import os
import glob
import time
import numpy as np
import pydicom
import nrrd
import SimpleITK as sitk
import matplotlib.pyplot as plt

from holoscan.core import Operator, OperatorSpec, Fragment

# Third-party 
from transforms.Orientation import orientation

########################################
# ReaderOperator
########################################
class ImageLoaderOperator(Operator):
    def __init__(
        self,
        fragment: Fragment, 
        *args,
        config, input_path, 
        **kwargs):
        self.meta = config
        self.input_path = input_path
        if '.dcm' in str(self.input_path):
            self._image_reader = 'pydicom'
        elif '.nrrd' in str(self.input_path):
            self._image_reader = 'nrrd'
        else: 
            self._image_reader = 'sitk'
            
        self.shape = (512, 512, 512)
        self.spacing = (1,1,1)
        self.origin = (0,0,0)
        self.direction = (1,0,0,0,1,0,0,0,1)
        self.input_name = "path"
        self.output_name = "image"
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output(self.output_name)

    def _nrrd_reader(self, input_path):
        img, header = nrrd.read(input_path)
        affine = header["space directions"]
        self.origin = header["space origin"]
        self.shape = img.shape
        # convert affine to SimpleITK style direction
        self.spacing = np.sqrt((affine ** 2).sum(axis=0))
        self.direction = (affine / self.spacing).flatten()
        img = orientation(
            img, transpose=self.meta["TRANSPOSE"][0]
        )	
        return img

    def _sitk_reader(self, input_path):
        img = sitk.ReadImage(input_path)
        self.spacing = img.GetSpacing()
        self.origin = img.GetOrigin()
        self.direction = img.GetDirection()
            
        input_data = sitk.GetArrayFromImage(img)
        self.shape = input_data.shape
        input_data = orientation(
            input_data, transpose=self.meta["TRANSPOSE"][0]
        )	
        return input_data

    def compute(self, op_input, op_output, context):
        input_data = self.input_path
        if not input_data:
            raise ValueError("Input image is not found.")

        patient_id = os.path.basename(input_data).split('.')[0]
        if self._image_reader == 'pydicom':
            input_data = self._pydicom_reader(input_data)
            transformed = False
        elif self._image_reader == 'nrrd':
            input_data = self._nrrd_reader(input_data)
            transformed = True
        else:
            input_data = self._sitk_reader(input_data)
            transformed = True

        # Construct metadata dictionary
        self.meta.update({
            "PatientID": patient_id,
            "InferenceDate": time.strftime("%Y-%m-%d"),
            "PixelSpacing": self.spacing,
            "Origin": self.origin,
            "Direction": self.direction,
            "TransformedFromLoader": transformed
        })
        # calculate affine 
        affine = self.calculate_affine_matrix()

        # Output dictionary for downstream operators
        output = {"image": input_data, "meta": self.meta, "affine": affine}
        op_output.emit(output, self.output_name)

    def calculate_affine_matrix(self):
        """
        Compute the 4x4 affine transformation matrix based on spacing, origin, and direction.
        """
        affine = np.eye(4)
        direction_matrix = np.array(self.direction).reshape(3, 3)

        # Apply spacing
        for i in range(3):
            affine[i, :3] = direction_matrix[i] * self.spacing[i]

        # Set the origin
        affine[:3, 3] = np.array(self.origin)

        return affine


########################################
# ImageSaverOperatoroutput_dir
########################################
class ImageSaverOperator(Operator):
    def __init__(
        self,
        fragment: Fragment, 
        *args,
        output_dir,
        **kwargs):
        self.output_dir = output_dir
        self.input_name = "image"
        super().__init__(fragment, *args, **kwargs)

    def saver(self, this_class_data, spacing, origin, direction, transformed, transpose, output_dir, class_name):
        if transformed:
            this_class_data = orientation(
                this_class_data, transpose=transpose
            )
        if this_class_data.dtype!=np.uint8:
            this_class_data = this_class_data.astype(np.uint8)

        img = sitk.GetImageFromArray(this_class_data)
        img.SetSpacing(spacing)
        img.SetOrigin(origin)
        img.SetDirection(direction)

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, class_name+'.nii.gz')
        sitk.WriteImage(img, output_path)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name)

    def compute(self, op_input, op_output, context):
        input_data = op_input.receive(self.input_name)
        if input_data is None:
            raise ValueError("Input image is not found.")

        for i, class_name in enumerate(input_data["meta"]["CLASSES"].values()):		
            self.saver(
                input_data["image"][i],
                input_data["meta"]["PixelSpacing"],
                input_data["meta"]["Origin"],
                input_data["meta"]["Direction"],
                input_data["meta"]["TransformedFromLoader"],
                input_data["meta"]["TRANSPOSE"][1],
                self.output_dir, class_name)

        print(f"✅ Image successfully saved at: {self.output_dir}")
        return True

