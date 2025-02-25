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
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.meta = config
		self.spacing = (1,1,1)
		self.origin = (0,0,0)
		self.direction = (1,0,0,0,1,0,0,0,1)

    def process(self, input_data=None):
        try:
            if not input_data or not os.path.exists(input_data):
                raise ValueError(f"Invalid input file path: {input_data}")

            patient_id = os.path.basename(input_data).split('.')[0]
            img = sitk.ReadImage(input_data)
            self.spacing = img.GetSpacing()
            self.origin = img.GetOrigin()
            self.direction = img.GetDirection()
                
            input_data = sitk.GetArrayFromImage(img)
            input_data = orientation(
                input_data, transpose=self.meta["TRANSPOSE"][0]
            )	
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
            return output
        except:
            return False

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
	def __init__(self, output_dir):
        self.output_dir = output_dir

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

	def process(self, input_data):
        try:
            output_dir = os.path.join(self.output_dir, input_data["meta"]["InferenceDate"], input_data["meta"]["PatientID"])
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir, exist_ok=True)

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
        except Exception as e:
            print(f"❌ Failed to save image: {e}")
            return False

########################################
# ResultDisplayOperator
########################################
class ResultDisplayOperator(Operator):
    """
    A simple result display operator that receives a dictionary containing segmentation results.
    It then displays or saves the segmentation using matplotlib.
    """
    def __init__(self, display_interval=1.0, **kwargs):
        super().__init__(**kwargs)
        self.display_interval = display_interval  # seconds between displays

    def process(self, input_data):
        # Expect input_data to be a dictionary containing "segmentation"
        if input_data is None or "segmentation" not in input_data:
            print("No segmentation output to display.")
            return None

        segmentation = input_data["segmentation"]
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
        # Optionally, you could also save the output to a file.
        return input_data

