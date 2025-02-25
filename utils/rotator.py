# -*- coding: utf-8 -*-
# Copyright 2023 by Jung-eun Park, Hutom.
# All rights reserved.

import math
import numpy as np 
from scipy.spatial.transform import Rotation
from scipy.ndimage import rotate

__all__ = [
    'deg2rad', 'rad2deg', 'check_rotation', 'rotate_forward'
]


def deg2rad(deg): return deg * math.pi / 180.0

def rad2deg(rad): return rad * 180.0 / math.pi

def cropND(img, z, y, x):
    diff = [(o-n)//2 for o, n in zip(img.shape, [z, y, x])]
    difz, dify, difx = diff
    return img[difz:z+difz, dify:y+dify, difx:x+difx]

def check_rotation(meta_info):
    cos1, _, sin1, _, cos2, sin2 = meta_info["ImageOrientation(Patient)"]
    theta1 = math.asin(sin1) if -1<= sin1 <= 1 else math.acos(cos1)
    theta2 = math.asin(sin2) if -1<= sin2 <= 1 else math.acos(cos2)
    theta1 = rad2deg(theta1)
    theta2 = rad2deg(theta2)
    if theta1 > 5.0 or theta2 > 5.0:
        meta_info["APPLY_TRANSFORM"] = True
        meta_info["Return_Angles"] = [-theta1, -theta2, 0]
        meta_info = rotate_forward(meta_info["Return_Angles"], meta_info, reshape=False)
        return meta_info
    else:
        return meta_info

def rotate_forward(rot_angles, meta_info, reshape=False, reverse=False):
    image = meta_info["PixelData"].copy()
    rows, cols = image.shape[1], image.shape[2]
    num_slices = image.shape[0] 
    if reshape==True:
        z_length = meta_info["SliceThickness"] * num_slices
        _,_,z = meta_info["new_yxz"]
        ratio = z_length / (max(z)-min(z))
        num_slices = math.ceil(num_slices * ratio)
        meta_info["SliceThickness"] = z_length / num_slices
    min_val = np.min(image)

    if reverse==True: rot_angles = -1 * np.array(rot_angles)

    if rot_angles[0] != 0: # x-axis rotation
        image = rotate(
            image, rot_angles[0], axes=(2,0), reshape=True, 
            mode='constant', cval=min_val
        )
    if rot_angles[1] != 0: # y-axis rotation
        image = rotate(
            image, rot_angles[1], axes=(1,0), reshape=True, 
            mode='constant', cval=min_val
        )
    if rot_angles[2] != 0: # z-axis flip
        image = image[::-1]

    image = cropND(image, num_slices, rows, cols)
    meta_info["ImageOrientation(Patient)"] = [1,0,0,0,1,0] 
    meta_info["Return_Angles"] = rot_angles
    meta_info["PixelData"] = image
    return meta_info