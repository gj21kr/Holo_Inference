from __future__ import annotations

from monai.transforms import ScaleIntensityRanged, EnsureChannelFirstd, Orientationd

from transforms.ImageProcessing import *
from utils.resampler import Resampler

__all__ = ['call_trans_function']

def check_config(config):
    original_shape = config["original_shape"]
    original_spacing = config["original_spacing"]
    target_shape = None
    target_spacing = None
    if config["SPACING"][0] is None and config["SPACING"][1] is not None:        
        target_spacing = [*config["original_spacing"][:2], config["SPACING"][-1]]
    elif config["SPACING"][0] is not None and config["SPACING"][1] is not None:
        target_spacing = config["SPACING"]
    elif config["SPACING"][0] is None and config["SPACING"][1] is None:
        target_spacing = config["original_spacing"]

    if config["TRANSPOSE"][0]==(0,1,2) or config["TRANSPOSE"][0]==(0,2,1):
        original_spacing = [original_spacing[-1], *original_spacing[:2]]
        target_spacing = [target_spacing[-1], *target_spacing[:2]]
    return original_spacing, original_shape, target_spacing, target_shape

def call_trans_function(config):
    test_transforms = []
    
    if "BoneMask" in config.keys() and config["BoneMask"][0]==True:
        test_transforms += [
            BoneMaskedArrayd(keys=["image"], min_val=config["CONTRAST"][0], 
                             phase=config["BoneMask"][1], pname=config["case_name"],
                             bone_folder=config["BoneMask"][2])
        ]
        
    int_norm = config["INT_NORM"].lower()
    if '+' in int_norm: 
        int_norm = int_norm.split('+')
    else:
        int_norm = [int_norm]
    for config_name in int_norm:
        if config_name in ['scale', 'windowing', 'clip']:
            test_transforms += [
                ScaleIntensityRanged(keys=["image"],
                    a_min=config["CONTRAST"][0], a_max=config["CONTRAST"][1], 
                    b_min=0, b_max=1, clip=True),
                ]
        elif config_name in ['z_norm', 'znorm', 'z norm']:
            test_transforms += [
                ZNormalizationd(keys=["image"],contrast=config["CONTRAST"])
            ]
        elif config_name in ['min_max_norm', 'norm', 'min max norm']:
            test_transforms += [
                Normalizationd(keys=["image"])
            ]
        # else:
        #     print('Not Intensity Normalization')
    if config["SPACING"] != [None,None,None]:
        original_spacing, original_shape, target_spacing, target_shape = check_config(config)
        resampler = Resampler(
            config["interp_mode"].split('_')[0], 
            original_spacing, target_spacing, 
            original_shape, target_shape, out_dtype='tensor')  
        test_transforms += [resampler]
    else:
        resampler = None
        
    if "ChannelDuplication" in config.keys() and config["ChannelDuplication"][0]==True:
        test_transforms += [
            DuplicateChanneld(keys=["image"], channel_dim=config["ChannelDuplication"][1])
        ]
    if "RnadMultiWindow" in config.keys() and config["RnadMultiWindow"][0]==True:
        test_transforms += [
            RandMultiWindowingd(keys=["image"], window=config["CONTRAST"], 
                                ranges=config["RnadMultiWindow"][1], output_nums=config["CHANNEL_IN"])
        ]
    if "TargetMultiWindow" in config.keys() and config["TargetMultiWindow"][0]==True:
        test_transforms += [
            TargetMultiWindowingd(keys=["image"], window=config["CONTRAST"], 
                                ranges=config["TargetMultiWindow"][1], output_nums=config["CHANNEL_IN"])
        ]
    if 'ChannelFirst' in config.keys() and config["ChannelFirst"]==True:
        test_transforms += [
                EnsureChannelFirstd(keys=["image"], channel_dim='no_channel'),
            ]
    return test_transforms, resampler