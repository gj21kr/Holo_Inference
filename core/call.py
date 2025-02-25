from __future__ import annotations

import os, sys

from monai.data import *

__all__=[
    "call_model"
 ]

def call_model(config):
    model = None
    model_name = config["MODEL_NAME"].lower()
    if model_name=='nnunet' or model_name=='ys_nnunet':
        from core.nnUNet import nnUNet
        model = nnUNet(
            in_channels = config["CHANNEL_IN"],
            num_classes = config["CHANNEL_OUT"],
            channels = config["MODEL_CHANNEL_IN"],
            deep_supervision = config["DEEP_SUPERVISION"],
        )
    elif model_name=='dilateddenseunet':
        from core.DilatedDenseUNet import DenseResUNet3D
        model = DenseResUNet3D(
            in_channels = config["CHANNEL_IN"],
            out_channels = config["CHANNEL_OUT"],
            base_filters = config["CHANNEL_LIST"][0],
            num_blocks = config["NUM_RES_UNITS"],
            growth_rate = config["GROWTH_RATE"],
            deep_supervision = config["DEEP_SUPERVISION"],
        )
    elif model_name=='denseunet':
        from core.DenseUNet import SkipDenseNet3D
        model = SkipDenseNet3D(
            in_channels = config["CHANNEL_IN"],
            classes = config["CHANNEL_OUT"],
            growth_rate = config["GROWTH_RATE"] if "GROWTH_RATE" in config.keys() else 16, 
            num_init_features = config["CHANNEL_LIST"][0],
            drop_rate = config["DROPOUT"]
        )
    elif 'ahnet' in model_name:
        from monai.networks.nets import AHNet 
        model = AHNet(
            in_channels = config["CHANNEL_IN"],
            out_channels = config["CHANNEL_OUT"],
            layers = config["layers"], 
            psp_block_num = 4,
            progress = False, 
            upsample_mode = 'transpose', 
            spatial_dims = 3
        )
    elif 'segresnet' in model_name:
        from monai.networks.nets import SegResNet 
        model = SegResNet(
            in_channels = config["CHANNEL_IN"],
            out_channels = config["CHANNEL_OUT"],
            spatial_dims = 3,
            init_filters = config["CHANNEL_LIST"][0],
            dropout_prob = config["DROPOUT"],
            upsample_mode = 'deconv',
        )
    elif 'highresnet' in model_name:
        from core.HighResNet import HighResNet
        model = HighResNet(
            in_channels = config["CHANNEL_IN"],
            out_channels = config["CHANNEL_OUT"],
            dimensions = 3,
            initial_out_channels_power=4,
            layers_per_residual_block=2,
            residual_blocks_per_dilation=3,
            dilations=3,
            residual=True,
            padding_mode=config["Padding"], # 'constant', 'reflect', 'replicate'
            add_dropout_layer=True,
            batch_norm= True if config["Norm_Type"]=="batch" else False,
            instance_norm=False if config["Norm_Type"]=="batch" else True, 
        )
    elif 'attentionunet' in model_name:
        from core.AttentionUNet import AttentionUnet
        model = AttentionUnet(
            spatial_dims = 3,
            in_channels = config["CHANNEL_IN"],
            out_channels = config["CHANNEL_OUT"],
            channels = config["CHANNEL_LIST"],
            strides = config["STRIDES"],
            dropout = 0,
        )

    assert model is not None, 'Model Error!'    
    return model
