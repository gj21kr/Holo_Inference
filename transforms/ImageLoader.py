import os 
# from monai.transforms import LoadImage 
import numpy as np
import SimpleITK as sitk

class ImageLoad():
	def __init__(self):
		self.spacing = (1,1,1)
		self.origin = (0,0,0)
		self.direction = (1,0,0,0,1,0,0,0,1)

	def __call__(self, input_data):
		img = sitk.ReadImage(input_data)
		input_data = sitk.GetArrayFromImage(img)
		self.spacing = img.GetSpacing()
		self.origin = img.GetOrigin()
		self.direction = img.GetDirection()
		return input_data

class ImageSave():
	def __init__(self, spacing, origin, direction):
		self.spacing = spacing
		self.origin = origin
		self.direction = direction

	def __call__(self, input_data, output_path):
		if input_data.ndim==4:
			input_data = input_data[0]
		if input_data.dtype!=np.uint8:
			input_data = input_data.astype(np.uint8)
			
		img = sitk.GetImageFromArray(input_data)
		img.SetSpacing(self.spacing)
		img.SetOrigin(self.origin)
		img.SetDirection(self.direction)
		sitk.WriteImage(img, output_path)