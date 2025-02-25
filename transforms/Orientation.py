from __future__ import annotations
import numpy as np
import gc

__all__ = [
	"orientation", "orientation_revert"	
]

def orientation(image, flipx=False, flipy=False, flipz=False, transpose=(0,1,2)):
	def do(image, flipx, flipy, flipz, transpose):
		if transpose!=(0,1,2) : image = np.transpose(image, transpose)
		if flipx : 
			new_image = np.empty_like(image)
			for i in range(image.shape[2]):
				new_image[:,:,i] = np.fliplr(image[:,:,i])
			image = new_image; del new_image; gc.collect()
		if flipy : 
			new_image = np.empty_like(image)
			for i in range(image.shape[2]):
				new_image[:,:,i] = np.flipud(image[:,:,i])
			image = new_image; del new_image; gc.collect()
		if flipz : 
			image = image[:,:,::-1]
		return image

	if image.ndim == 3: 
		image = do(image, flipx, flipy, flipz, transpose)
	elif image.ndim == 4:
		for c in range(image.shape[0]):
			image[c] = do(image[c], flipx, flipy, flipz, transpose)
	return image

def orientation_revert(image, flipx=False, flipy=False, flipz=False, transpose=(0,1,2)):
	def do(image, flipx, flipy, flipz, transpose):
		if flipx : 
			new_image = np.empty_like(image)
			for i in range(image.shape[2]):
				new_image[:,:,i] = np.fliplr(image[:,:,i])
			image = new_image; del new_image; gc.collect()
		if flipy : 
			new_image = np.empty_like(image)
			for i in range(image.shape[2]):
				new_image[:,:,i] = np.flipud(image[:,:,i])
			image = new_image; del new_image; gc.collect()
		if flipz : 
			image = image[:,:,::-1]
		if transpose!=(0,1,2) : image = np.transpose(image, transpose)
		return image

	if image.ndim == 3: 
		image = do(image, flipx, flipy, flipz, transpose)
	elif image.ndim == 4:
		new_image = []
		for c in range(image.shape[0]):
			new_image.append(do(image[c], flipx, flipy, flipz, transpose))
		image = np.stack(new_image, axis=0)
	return image
