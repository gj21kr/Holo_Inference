# -*- coding: utf-8 -*-
# Copyright 2023 by Jung-eun Park, Hutom.
# All rights reserved.
import gc
import torch
import numpy as np
from joblib import Parallel, delayed
from monai.data.meta_tensor import MetaTensor
from monai.transforms import (
        Compose, RandAffined, EnsureChannelFirstd
)
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from core.call import call_model

from utils.inferer import SlidingWindowInferer
from utils.utils import load_saved_model, add_array

from transforms.Orientation import orientation
from transforms.call_preproc import call_trans_function, check_config

job_threshold = 2
affine = torch.as_tensor([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1]], dtype=torch.float64)

class do_inference():
    def __init__(self, config, model, raw, ensemble:int=0):
        self.original_spacing, self.original_shape, self.target_spacing, self.target_shape = check_config(config)

        self.add_channel = False
        self.invert = False
        self.active = None
        
        if "interp_mode" in config.keys():
            self.interp_mode = config["interp_mode"]
        else:
            self.interp_mode = "trilinear_trilinear"
  
        if "AMP" in config.keys() and config["AMP"]==True:
            self.amp = True
            self.dtype = torch.float16
        else:
            self.amp = False
            self.dtype = torch.float32

        self.test_transforms, self.resampler = call_trans_function(config)
        
        if config["MODE"] is not None and config["MODE"].lower() == 'tta':
            self.invert = True 

        self.model = model[ensemble] if type(model) is list else model
        self.weight = config["WEIGHTS"][ensemble]

        if "DEEP_SUPERVISION" in config.keys() and config["DEEP_SUPERVISION"]==True:
            self.deep_supervision = True
        else: 
            self.deep_supervision = False

        if "ACTIVATION" in config.keys():
            if type(config["ACTIVATION"]) is list:
                if config["ACTIVATION"][ensemble].lower()=='sigmoid':
                    self.active=torch.nn.Sigmoid()
                elif config["ACTIVATION"][ensemble].lower()=='softmax':
                    self.active=torch.nn.Softmax(dim=0)
            else:
                if config["ACTIVATION"].lower()=='sigmoid':
                    self.active=torch.nn.Sigmoid()
                elif config["ACTIVATION"].lower()=='softmax':
                    self.active=torch.nn.Softmax(dim=0)
                    
        channel_out = config["CHANNEL_OUT"]
        if channel_out > len(config["CLASSES"].keys()):
            self.include_background = False
        else:
            self.include_background = True

        self.inferer = SlidingWindowInferer(
            roi_size=config["INPUT_SHAPE"],
            sw_batch_size=config["BATCH_SIZE"],
            sw_device=torch.device("cuda"),
            device=torch.device("cpu"),
            overlap=0.5,
            deep_supervision=self.deep_supervision
        )
        self.input_image = raw
        
    def data_prepare(self, config):
        ## Preprocessing for the inference-framework difference..         
        if "HU_WEIGHTING" in list(config.keys()) and config["HU_WEIGHTING"][0]:
            config["NEW_CH"] = self.load_new_channel_data(
                config, config["HU_WEIGHTING"][1], config["HU_WEIGHTING"][2], config["APPLY_TRANSFORM"])
            config["NEW_CH"] = orientation(
                config["NEW_CH"], config["FLIP_XYZ"][0], config["FLIP_XYZ"][1], 
                config["FLIP_XYZ"][2], config["TRANSPOSE"][0]
                )
            self.input_image[config["NEW_CH"]>0] = self.input_image[config["NEW_CH"]>0] * config["HU_WEIGHTING"][3]

            self.input_image = {"image":MetaTensor(self.input_image,affine=affine)}            
        elif "ADD_INPUT_CH" in list(config.keys()) and config["ADD_INPUT_CH"][0]==True:
            config["NEW_CH"] = self.load_new_channel_data(
                config, config["ADD_INPUT_CH"][1], config["ADD_INPUT_CH"][2], config["APPLY_TRANSFORM"])
            config["NEW_CH"] = orientation(
                config["NEW_CH"], config["FLIP_XYZ"][0], config["FLIP_XYZ"][1],
                config["FLIP_XYZ"][2], config["TRANSPOSE"][0])

            self.input_image = {
                "image":MetaTensor(self.input_image,affine=affine), 
                "mask":MetaTensor(config["NEW_CH"],affine=affine)}
        else:
            self.input_image = {"image":MetaTensor(self.input_image,affine=affine)}

    def pre(self):
        test_transforms = self.test_transforms
        if self.invert==True:
            test_transforms += [
                EnsureChannelFirstd(list(self.input_image.keys()),channel_dim='no_channel'),
                RandAffined(
                    keys=list(self.input_image.keys()), prob=1.0, 
                    spatial_size=self.input_image["image"].shape,
                    rotate_range=(-0.1,0.1), 
                    shear_range=(-0.1,0.1), 
                    translate_range=None, 
                    scale_range=(0.8,1.2),
                    mode=['bilinear'],
                    padding_mode='zeros',
                    cache_grid=True
                )
            ]
        self.transforms = Compose(test_transforms)


    def post(self, data):
        if self.invert==True:
            data = self.transforms.inverse(data)

        if self.include_background==False:
            data["image"] = data["image"][1:]

        if self.resampler is not None:
            data["image"] = self.resampler(
                data["image"], revert=True, out_dtype='ndarray', with_channels=True, device='cpu')
            return data["image"]
        else:			
            return data["image"].cpu().numpy()
        
    @torch.no_grad()
    def inference(self):
        data = self.transforms(self.input_image)
        with torch.autocast(enabled=self.amp, dtype=self.dtype, device_type='cuda'):	
            if "mask" in list(data.keys()):
                data["image"] = torch.reshape(data["image"],(1, *data["image"].shape[-3:]))
                m = data["mask"]
                m = torch.reshape(m,(1, *m.shape[-3:]))	
                data["image"] = torch.stack([data["image"], m], dim=1)
            else:
                if len(data["image"].shape)==3:
                    data["image"] = torch.reshape(data["image"],(1, 1, *data["image"].shape))
                elif len(data["image"].shape)==4:
                    data["image"] = torch.reshape(data["image"],(1, *data["image"].shape[-3:]))
            data["image"] = self.inferer(inputs=data["image"].to(self.dtype), network=self.model)[0]  * self.weight
            data["image"] = self.active(data["image"]) if not self.active is None else data["image"]
            torch.cuda.empty_cache()
            return data

def post_inference(config, image, results, post_transform):	
    if config["ARGMAX"]==True:
        results = results.astype(np.float16)
        bg = np.sum(np.stack(
                [(r<config["THRESHOLD"]).astype(np.uint8) for r in results], axis=0
            ), axis=0, keepdims=True, dtype=np.uint8)/len(results)
        results = np.concatenate((bg,results),axis=0,dtype=np.float16)
        argmax = np.argmax(results, axis=0)
        results = np.eye(results.shape[0])[...,argmax]
        results = results[1:].astype(np.uint8)
        
    num_classes, save_classes = 0, 0
    if "SAVE_CLASSES" in config.keys():
        num_classes = len(config["SAVE_CLASSES"])
        save_classes = [config["CLASSES"][i] for i in config["SAVE_CLASSES"]]
        t = [results[i-1] for i in config["SAVE_CLASSES"]]
        results = np.stack(t,axis=0)
    else:
        num_classes = len(config["CLASSES"].keys())
        save_classes = list(config["CLASSES"].values())
 
    def process_index(i, this_class, this_image, image, config, post_transform):
        if len(post_transform)>0:
            for func in post_transform:
                if func.apply_labels is None or i in func.apply_labels: 
                    if func.image_require==True :
                        this_image = func(this_image, image)
                    elif func.image_require=='mask':
                        this_image = func(this_image, config["NEW_CH"])
                    else:
                        this_image = func(this_image)
        return this_image
    results = [process_index(
        i, save_classes[i], results[i], image, config, post_transform) for i in range(num_classes)]
    return np.stack(results, axis=0)

def main(configs, input_array, post_transform) -> None:
    # Inference	
    if isinstance(configs, list):
        num_ensemble = len(configs) 
    else:
        num_ensemble = 1
        configs = [configs]

    results = None
    for i in range(num_ensemble):
        config = configs[i]
        if config["WEIGHTS"] == 0 : continue
        model = load_saved_model(config, call_model(config))
        inferer = do_inference(config, model, input_array, i)
        inferer.data_prepare(config)
        inferer.pre()
        num_tta = 3 if config["MODE"] is not None and 'tta' in config["MODE"].lower() else 1
        for _ in range(num_tta):
            data = inferer.inference()
            data = inferer.post(data)
            results = add_array(results, data)
        del data, inferer; gc.collect()
        torch.cuda.empty_cache()
    del model; gc.collect()
    torch.cuda.empty_cache()
 
    if num_ensemble*num_tta > 1:
        results = results / (num_ensemble*num_tta)

    # Postprocessing & Save Results
    results = post_inference(config, input_array, results, post_transform)
    return results