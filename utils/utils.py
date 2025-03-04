# -*- coding: utf-8 -*-
# Copyright 2023 by Jung-eun Park, Hutom.
# All rights reserved.
import os, gc, glob
import shutil
import argparse as ap
import SimpleITK as sitk

import numpy as np 
import torch 
from joblib import Parallel, delayed

from core.call import call_model
from transforms.Orientation import orientation, orientation_revert
from utils.reorientation import image_affine_reorientation

__all__ = [
    'add_array', 'sum_list_of_array', 'load_image',
    'load_saved_model', 'arg2target', 'str2bool',
    'merger', 'dict_update', 'revert_orientation',
    "RemoveSamllObjects", 'gen_models', 'saver', 'nifti_rename'
]
class RemoveSamllObjects:
    def __init__(self, min_size=1e03, connectivity=3, apply_labels=None):
        import skimage
        self.morphology = skimage.morphology
        self.connectivity = connectivity
        self.minSz = min_size
        self.apply_labels = apply_labels
        self.image_require = False
        import numpy as np
        self.lib = np

    def __call__(self, mask):
        if type(mask)==list: mask = self.lib.array(mask)

        def work(m):
            m = self.morphology.remove_small_objects(
                m.astype(bool), min_size=self.minSz, connectivity=self.connectivity)
            return m
        return work(mask).astype(self.lib.uint8)	

def sitk_nifti_reader(file):
    try:
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(file)
        image = reader.Execute()
    except:
        image = sitk.ReadImage(file, imageIO="NiftiImageIO")
    return image

def sitk_nifti_writer(image, file):
    try:
        writer = sitk.ImageFileWriter()
        writer.SetImageIO("NiftiImageIO")
        writer.SetFileName(file)
        writer.UseCompressionOn()
        writer.Execute(image)
    except:
        try:
            sitk.WriteImage(image, file, imageIO="NiftiImageIO")
        except:
            print('Cannot write this file!', file)
            
def nifti_rename(output_path, njobs=3):
    def worker(target):  
        target_name = target.split('/')[-1]
        target_folder = target.split(target_name)[0]
        target_new = target_name.capitalize()
        if 'iliopsoas' in target:
            target_new = target_new.replace('Iliopsoas','Psoas_muscle')
        if 'inferior_vena_cava' in target:
            target_new = target_new.replace('Inferior_vena_cava','IVC')
        if '_left' in target:
            target_new = target_new.replace('left',"L")
        if '_right' in target:
            target_new = target_new.replace('right',"R")
        nfile = os.path.join(output_path, target_new)
        shutil.move(target, nfile)
        return
        
    output_files = glob.glob(os.path.join(output_path,'*.nii.gz'))
    if len(output_files)<3:
        njobs = len(output_files)
    Parallel(n_jobs=njobs)(delayed(worker)(target) for target in output_files)

def mask_subtract(folder, target='', keep='', value=255.):
    if not '.nii.gz' in target: 
        target = target+'.nii.gz'
    if not '.nii.gz' in keep: 
        keep = keep+'.nii.gz'
    target_path = os.path.join(folder, target)
    keep_path = os.path.join(folder, keep)
    if not os.path.isfile(keep_path):
        print(f'{keep} should be predicted first!')
        return 
    target = sitk_nifti_reader(target_path)
    keep = sitk_nifti_reader(keep_path)
    target_arr = sitk.GetArrayFromImage(target)
    keep_arr = sitk.GetArrayFromImage(keep).transpose(0,2,1)[:,::-1]
    target_arr[keep_arr>0] = 0
    target = sitk.GetImageFromArray(target_arr.astype(np.uint8))
    target.SetSpacing(keep.GetSpacing())
    target.SetOrigin(keep.GetOrigin())
    target.SetDirection(keep.GetDirection())
    sitk_nifti_writer(target, target_path)

def mask_merger(folder, ins, out, value=255.):
    if not '.nii.gz' in out: 
        out = out+'.nii.gz'
        
    image = None
    for ind, in_ in enumerate(ins):
        if not '.nii.gz' in in_:
            in_ = in_+'.nii.gz'
            
        input_path = os.path.join(folder, in_)
        if ind ==0:
            image = sitk_nifti_reader(input_path)
            spacing = image.GetSpacing()
            origin = image.GetOrigin()
            direction = image.GetDirection()
            image = (sitk.GetArrayFromImage(image)>0).astype(np.uint8)
        else:
            image[sitk.GetArrayFromImage(sitk_nifti_reader(input_path))>0] = 1
            
    image = sitk.GetImageFromArray((image*value).astype(np.uint8))
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    sitk_nifti_writer(image, os.path.join(folder, out))
            
def revert_orientation(config, output_path, njobs=3):
    def worker(file, config):
        image = sitk_nifti_reader(file)
        image = sitk.GetArrayFromImage(image)
        if "Reorientator" in list(config.keys()):
            image = config["Reorientator"].RAS_array_Reverse(image)

        image = sitk.GetImageFromArray(image)
        image.SetSpacing(config['original_spacing'])
        image.SetOrigin(config['original_origin'])
        image.SetDirection(sitk_dire(config["original_direction"]))
        sitk_nifti_writer(image, file)
    
    output_files = glob.glob(os.path.join(output_path,'*.nii.gz'))
    # print('Converted to RSA->', config["Reorientator"].axcode)
    if len(output_files)<3:
        njobs = len(output_files)
    Parallel(n_jobs=njobs)(delayed(worker)(file, config) for file in output_files)
    # [worker(file, config) for file in output_files]
    
def add_array(arr1, arr2):
    if arr1 is None: 
        return arr2
    else:
        return np.add(arr1, arr2)

def sum_list_of_array(list1):
    for i, arr in enumerate(list1):
        if i ==0 : 
            result = arr
        else:
            result = np.add(result, arr)
    return result 

def load_image_nifti(configs, data_file):
    def get_affine(spacing, origin, direction):
        affine_matrix = np.zeros((4, 4))
        for i in range(3):
            for j in range(3):
                affine_matrix[i, j] = direction[i * 3 + j] * spacing[j]
            affine_matrix[i, 3] = origin[i]
        affine_matrix[3,3] = 1.
        return affine_matrix    
    # Data Load
    if os.path.isfile(data_file) is False: 
        print("Input Path should be the path of a file\n\t", data_file); return configs, None
    if isinstance(configs, list):
        config = configs[0]
    else:
        config = configs
    
    image = sitk_nifti_reader(data_file) 
    config["Reorientator"] = image_affine_reorientation(
                                get_affine(image.GetSpacing(),image.GetOrigin(),image.GetDirection()))
    raw = {}
    raw["PixelData"] = config["Reorientator"].img_to_RAS(sitk.GetArrayFromImage(image).astype(np.float32))
    raw["PixelData"] = orientation(
        raw["PixelData"], config["FLIP_XYZ"][0], config["FLIP_XYZ"][1], 
        config["FLIP_XYZ"][2], config["TRANSPOSE"][0])
    
    config["original_shape"] 		= raw["PixelData"].shape
    config["original_spacing"] 		= image.GetSpacing()
    config["original_origin"] 		= image.GetOrigin()
    config["original_direction"]	= image.GetDirection()
    
    splits = data_file.split('/') 
    config["series_name"] 			= splits[-2]
    config["case_name"] 			= splits[-1]
    if '.nii.gz' in config["case_name"]: config["case_name"] = config["case_name"].split('.nii.gz')[0]
    print('P_Name:', config["case_name"], 'Phase:', config["series_name"])
    config["rst_path"] = os.path.join(config["output_dir"], config["case_name"])
    config["rst_path"] = os.path.join(config["rst_path"], config["series_name"])
    if not os.path.isdir(config["rst_path"]): 
        os.makedirs(config["rst_path"], exist_ok=True); os.chmod(config["rst_path"], 0o777)

    configs = dict_update(configs, config)    
    return configs, raw

def load_image(configs, data_dir, output_type=None):
    if '@eaDir' in data_dir: return config, None
    if 'cache' in data_dir: return config, None
    if isinstance(configs, list):
        config = configs[0]
    else:
        config = configs
        
    # Data Load
    if os.path.isdir(data_dir) is False: 
        print("Input Path should be the path of a directory\n\t", data_dir); return config, None

    if "to_rsa" in list(config.keys()) and config["to_rsa"]==False:
        do_rsa = False
    else:
        do_rsa = True 

    if "do_transform" in list(config.keys()) and config["do_transform"]==False:
        do_ = False
    else:
        do_ = True
        config["do_transform"] = True
        
    if "rot_angles" in list(config.keys()) and config["rot_angles"]!=[0,0,0]:
        rot_angles = config["rot_angles"]
    else:
        rot_angles = [0,0,0]

    ct = Hutom_DicomLoader(
        data_dir, do_transform=do_, do_rsa=do_rsa,
        rot_angles=rot_angles, save_new_dicom=False)
    if ct.data is None: return config, None, None
    ct.data["PixelData"] = orientation(
        ct.data["PixelData"], config["FLIP_XYZ"][0], config["FLIP_XYZ"][1], 
        config["FLIP_XYZ"][2], config["TRANSPOSE"][0])
    
    config["original_shape"] 		= ct.data["PixelData"].shape
    config["original_spacing"] 		= ct.data["original_spacing"]
    config["original_origin"] 		= ct.data["original_origin"]
    config["original_direction"]	= ct.data["original_direction"]
    config["series_name"] 			= ct.data["series_name"]
    config["case_name"] 			= ct.data["PatientID"]

    if 'Return_Angles' in ct.data.keys():
        config["Return_Angles"]		= ct.data["Return_Angles"]
    if 'Reorientator' in ct.data.keys():
        config["Reorientator"]      = ct.data["Reorientator"]
    
    config["rst_path"] = os.path.join(config["output_dir"], config["case_name"])
    config["rst_path"] = os.path.join(config["rst_path"], config["series_name"])
    if not os.path.isdir(config["rst_path"]): 
        os.makedirs(config["rst_path"], exist_ok=True); os.chmod(config["rst_path"], 0o777)


    configs = dict_update(configs, config)
    return configs, ct.data

def load_saved_model(config, model):
    saved_model = ''
    if "MODEL_KEY" not in config.keys():
        key = "model_state_dict"
    else:
        key = config["MODEL_KEY"]
    if "DataParallel" not in config.keys():
        dp = True
    else:
        dp = config["DataParallel"]

    # saved_model = config["SAVED_MODEL"] if '.pth' in config["SAVED_MODEL"] else config["SAVED_MODEL"]+'.pth'
    saved_model = f'models/{config["MODEL_VERSION"]}.pth'
    if dp == True:
        model = torch.nn.DataParallel(model)
    # check cuda available 
    if torch.cuda.is_available():
        config["device"] = torch.device("cuda")
    else:
        config["device"] = torch.device("cpu")
    model_load = torch.load(saved_model, map_location=config["device"], weights_only=False)
    model.load_state_dict(model_load[key], strict=True)
    model.to(config["device"])
    model.eval()
    return model

def arg2targets(arg_t):
    if ',' not in arg_t and ' ' not in arg_t:
        print('This is multi-projects inference code. You can use other code for single target inference.')
    if ',' in arg_t:
        return arg_t.split(',')
    if ' ' in arg_t:
        return arg_t.split(' ')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ap.ArgumentTypeError('Boolean value expected.')


def sitk_dire(direction):
    # direction: (*IOP,*normal)
    return tuple([direction[i] for i in [0,3,6,1,4,7,2,5,8]])


def saver(class_, x, config, max_intensity=255) -> None: 
    # if len(np.unique(x))==1:
    #     print('No inference result!', class_)
    #     return 
    rst_path = config["rst_path"]
    x = orientation_revert(
        x, config["FLIP_XYZ"][0], config["FLIP_XYZ"][1], 
        config["FLIP_XYZ"][2], config["TRANSPOSE"][1])
    if 'Return_Angles' in config.keys():
        from utils.dcm_reader import rotate_forward
        config["PixelData"] = (x>0).astype(np.uint8)
        config = rotate_forward(
            rot_angles= config["Return_Angles"], 
            meta_info=config, reshape=False, reverse=True
        )
        x = config["PixelData"]
        
    if "ProbSave" in config.keys() and config["ProbSave"]==True:
        x = x.astype(float); #print(np.min(x), np.max(x))
        filename = f'{class_}_prob.nii.gz'
    else:
        x = ((x>0).astype(np.uint8)*max_intensity).astype(np.uint8)
        filename = f'{class_}.nii.gz'
    # print(config["original_spacing"])
    # print(config["original_origin"])
    # print(config["original_direction"])
    # print(sitk_dire(config["original_direction"]))
    x = sitk.GetImageFromArray(x)
    x.SetSpacing(config["original_spacing"])
    x.SetOrigin(config["original_origin"])
    x.SetDirection(sitk_dire(config["original_direction"]))
    
    writer = sitk.ImageFileWriter()
    save_img: str = os.path.join(rst_path, filename)
    writer.SetImageIO("NiftiImageIO")
    writer.SetUseCompression(True)
    writer.SetFileName(save_img)
    writer.Execute(x)
    os.chmod(save_img , 0o777)
    revert_orientation(config, rst_path)
    print("Saved",save_img)
    del x; gc.collect()

    
def merger(classes, config):
    rst_path = config["rst_path"]
    file_ = os.path.join(rst_path, f'{classes[1]}.nii.gz')
    img_ = sitk_nifti_reader(file_)
    arr_ = sitk.GetArrayFromImage(img_)
    for i in range(len(classes)):
        file_ = os.path.join(rst_path, f'{classes[i+1]}.nii.gz')
        img_ = sitk_nifti_reader(file_)
        temp_ = sitk.GetArrayFromImage(img_)
        np.putmask(arr_, temp_>0, i+1)
    img_ = sitk.GetImageFromArray(arr_.astype(np.uint8))
    img_.SetSpacing(config["original_spacing"])
    img_.SetOrigin(config["original_origin"])
    img_.SetDirection(sitk_dire(config["original_direction"]))
    save_img: str = os.path.join(rst_path, f'Merged.nii.gz')
    sitk_nifti_writer(img_, save_img)
    del temp_, img_, arr_; gc.collect()

def dict_update(configs, common):
    # config["original_shape"] = common["original_shape"]
    # config["original_spacing"] = common["original_spacing"]
    # config["original_origin"] = common["original_origin"]
    # config["original_direction"] = common["original_direction"]
    # config["case_name"] = common["case_name"]
    # config["series_name"] = common["series_name"]
    # config["rst_path"] = common["rst_path"]

    if isinstance(configs, list):
        for config in configs:
            for key, val in common.items():
                config[key] = val
    elif isinstance(configs, dict):
        for key, val in common.items():
            configs[key] = val        
    return configs

def gen_models(config):
    if config["MODE"] is not None and config["MODE"].lower() in ['ensemble']:
        models = []; new_configs = []; data_configs = {}
        for i in range(len(config["SAVED_MODEL"])):
            new_config = config.copy()
            for key in config.keys():
                if type(config[key]) is list and key not in [
                    "FLIP_XYZ", "TRANSPOSE", "rot_angles", "do_transform", "MODE", "THRESHOLD",
                    "ARGMAX", "SAVE_CT", "SAVE_MERGE", "from_raid", "CLASSES"
                    ]:
                    new_config[key] = config[key][i]
            models.append(call_model(new_config))
            new_configs.append(new_config)
        return models, new_configs
    else: # config["MODE"] is not None and config["MODE"].lower() in ['tta']
        new_config = config.copy()
        if type(new_config["CHANNEL_IN"]) is list and type(config["CHANNEL_IN"][0]) is int: 	
            new_config["CHANNEL_IN"] = config["CHANNEL_IN"][0]
        if type(new_config["CHANNEL_OUT"]) is list and type(config["CHANNEL_OUT"][0]) is int: 	
            new_config["CHANNEL_OUT"] = config["CHANNEL_OUT"][0]
        if type(new_config["MODEL_NAME"]) is list and type(config["MODEL_NAME"][0]) is str: 	
            new_config["MODEL_NAME"] = config["MODEL_NAME"][0]
        if type(new_config["SPACING"]) is list and type(config["SPACING"][0]) is list: 	
            new_config["SPACING"] = config["SPACING"][0]
        if type(new_config["INPUT_SHAPE"]) is list and type(config["INPUT_SHAPE"][0]) is list: 	
            new_config["INPUT_SHAPE"] = config["INPUT_SHAPE"][0]
        if type(new_config["CONTRAST"]) is list and type(config["CONTRAST"][0]) is list: 	
            new_config["CONTRAST"] = config["CONTRAST"][0]
        if "FEATURE_SIZE" in list(config.keys()):
            new_config["FEATURE_SIZE"] = config["FEATURE_SIZE"][0]
        if "PATCH_SIZE" in list(config.keys()) and type(config["PATCH_SIZE"][0]) is list:
            new_config["PATCH_SIZE"] = config["PATCH_SIZE"][0]
        return [call_model(new_config)], [new_config]