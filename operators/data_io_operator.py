import os
import glob
import time
import numpy as np
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt

# Holoscan operator base class import (adjust based on your Holoscan SDK)
from holoscan.core import Operator

# Third-party 
from transforms.Orientation import orientation

########################################
# NIFTIReaderOperator
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
        if '.dcm' in self.input_path:
            self._image_reader = 'pydicom'
        else: 
            self._image_reader = 'sitk'

		self.spacing = (1,1,1)
		self.origin = (0,0,0)
		self.direction = (1,0,0,0,1,0,0,0,1)
        self.input_name = "path"
        self.output_name = "image"
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output(self.output_name)

    def _sitk_reader(self, input_path):
        img = sitk.ReadImage(input_data)
        self.spacing = img.GetSpacing()
        self.origin = img.GetOrigin()
        self.direction = img.GetDirection()
            
        input_data = sitk.GetArrayFromImage(img)
        input_data = orientation(
            input_data, transpose=self.meta["TRANSPOSE"][0]
        )	
        return input_data

    def compute(self, op_input, op_output, context):
        input_image = self.input_path
        if not input_image:
            raise ValueError("Input image is not found.")

        patient_id = os.path.basename(input_data).split('.')[0]
        if self._image_reader == 'pydicom':
            input_data = self._pydicom_reader(input_data)
        else:
            input_data = self._sitk_reader(input_data)

        # Construct metadata dictionary
        self.meta.update({
            "PatientID": patient_id,
            "InferenceDate": time.strftime("%Y-%m-%d"),
            "PixelSpacing": self.spacing,
            "Origin": self.origin,
            "Direction": self.direction
        })
        # calculate affine 
        affine = self.calculate_affine_matrix()

        # Output dictionary for downstream operators
        output = {"image": image, "meta": meta, "transform": self.transform, "affine": affine}
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

    def saver(self, this_class_data, transpose, spacing, origin, direction, output_dir, class_name):
        this_class_data = orientation(
            this_class_data, transpose=transpose
        )	
        if this_class_data.ndim==4:
            this_class_data = this_class_data[0]
        if this_class_data.dtype!=np.uint8:
            this_class_data = this_class_data.astype(np.uint8)
            
        img = sitk.GetImageFromArray(this_class_data)
        img.SetSpacing(spacing)
        img.SetOrigin(origin)
        img.SetDirection(direction)

        output_path = os.path.join(output_dir, class_name+'.nii.gz')
        sitk.WriteImage(img, output_path)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name)

    def compute(self, op_input, op_output, context):
        input_data = op_input.receive(self.input_name)
        if not input_data:
            raise ValueError("Input image is not found.")

        for i, class_name in enumerate(input_data["meta"]["CLASSES"].values()):		
            self.saver(
                input_data["image"][i],
                input_data["meta"]["TRANSPOSE"][1],
                input_data["meta"]["PixelSpacing"],
                input_data["meta"]["Origin"],
                input_data["meta"]["Direction"],
                output_dir, class_name)

        print(f"✅ Image successfully saved at: {output_path}")
        return True


########################################
# ResultDisplayOperator
########################################
class ResultDisplayOperator(Operator):
    """
    A simple result display operator that receives a dictionary containing segmentation results.
    It then displays or saves the segmentation using matplotlib.
    """
    def __init__(
        self,
        fragment: Fragment, *args, display_interval=1.0, **kwargs):
        self.display_interval = display_interval  # seconds between displays
        self.input_name = "segmentation"
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name)

    def compute(self, op_input, op_output, context):
        # Expect input_data to be a dictionary containing "segmentation"
        segmentation = op_input.receive(self.input_name)
        if not segmentation:
            raise ValueError("Input segmentation is not found.")

        # Assume segmentation is a 2D array (if 3D, you may select a slice)
        if segmentation.ndim == 3:
            # For example, select the middle slice along the depth axis:
            slice_idx = segmentation.shape[0] // 2
            segmentation_to_show = segmentation[slice_idx]
        else:
            segmentation_to_show = segmentation

        plt.figure(figsize=(6, 6))
        plt.imshow(segmentation_to_show, cmap="gray")
        plt.title("Segmentation Result")
        plt.axis("off")
        plt.show(block=False)
        plt.pause(self.display_interval)
        plt.close()
