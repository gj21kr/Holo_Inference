'''
author:        lhssu <lhssu@hutom.co.kr>
date:          2024-02-05 13:09:22
Copyright © Hutom All rights reserved
'''
"""
why pydicom_yxz
dicom load array order
pydicom: z-axis transpose from zyx --> yxz

array & affine-matrix
* y:R&L; 
* x:A&P; 
* z:S&I.



"""
import nibabel as nib

class image_affine_reorientation:
    def __init__(self, original_affine,pydicom_yxz=True):
        self.pydicom_yxz=pydicom_yxz
        self.original_affine = original_affine
    
    def affine_to_RAS(self):
        axcode = nib.aff2axcodes(self.original_affine)
        for value in axcode:
            if value in ('R','L'):
                list1 = axcode.index(value)
            if value in ('P','A'):
                list2 = axcode.index(value)
            if value in ('S','I'):
                list3 = axcode.index(value)
        reorder_affine = self.original_affine[:,[list1,list2,list3,3]]
        reorder_axcode = nib.aff2axcodes(reorder_affine)
        for value in reorder_axcode:
            if value != 'R':
                reorder_affine[:,0] = reorder_affine[:,0]*-1
            if value != 'A':
                reorder_affine[:,1] = reorder_affine[:,1]*-1
            if value != 'S':
                reorder_affine[:,2] = reorder_affine[:,2]*-1

        assert nib.aff2axcodes(reorder_affine)==('R', 'A', 'S')
        return reorder_affine
    def img_to_RAS(self,data_array):
        if self.pydicom_yxz:
            data_array = data_array.transpose([1,2,0])
        axcode = nib.aff2axcodes(self.original_affine)
        for value in axcode:
            if value in ('R','L'):
                list1 = axcode.index(value)
            if value in ('P','A'):
                list2 = axcode.index(value)
            if value in ('S','I'):
                list3 = axcode.index(value)
        reorder_affine = self.original_affine[:,[list1,list2,list3,3]]
        data_array_trans = data_array.transpose([list1,list2,list3])
        reorder_axcode = nib.aff2axcodes(reorder_affine)
        #for value in reorder_axcode:
        if reorder_axcode[0] != 'R':
            reorder_affine[:,0] = reorder_affine[:,0]*-1
            data_array_trans = data_array_trans[::-1,:,:]
        if reorder_axcode[1] != 'A':
            reorder_affine[:,1] = reorder_affine[:,1]*-1
            data_array_trans = data_array_trans[:,::-1,:]
        if reorder_axcode[2] != 'S':
            reorder_affine[:,2] = reorder_affine[:,2]*-1
            data_array_trans = data_array_trans[:,:,::-1]
        assert nib.aff2axcodes(reorder_affine)==('R', 'A', 'S')
        if self.pydicom_yxz:
            data_array_trans = data_array_trans.transpose([2,0,1])
        return data_array_trans
    def RAS_array_Reverse(self,ras_array):
        if self.pydicom_yxz:
            ras_array = ras_array.transpose([1,2,0])
        axcode = nib.aff2axcodes(self.original_affine)
        for value in axcode:
            if value in ('R','L'):
                list1 = axcode.index(value)
            if value in ('P','A'):
                list2 = axcode.index(value)
            if value in ('S','I'):
                list3 = axcode.index(value)
        reorder_affine = self.original_affine[:,[list1,list2,list3,3]]
        reorder_axcode = nib.aff2axcodes(reorder_affine)
        if reorder_axcode[0] != 'R':
            ras_array = ras_array[::-1,:,:]
        if reorder_axcode[1] != 'A':
            ras_array = ras_array[:,::-1,:]
        if reorder_axcode[2] != 'S':
            ras_array = ras_array[:,:,::-1]
        data_array_trans = ras_array.transpose([list1,list2,list3])
        if self.pydicom_yxz:
            data_array_trans = data_array_trans.transpose([2,0,1])
        return data_array_trans