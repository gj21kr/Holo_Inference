import torch
import monai
import nibabel as nib
import numpy as np
import os
import vtk

# Import MONAI functions 
from monai.transforms import (
        Compose, RandAffined, EnsureChannelFirstd
)

# Import Holoscan base operator (the exact import may vary based on Holoscan version)
from holoscan.core import Operator  # Example; adjust based on actual Holoscan SDK

# Third-party
from core.call import call_model
from utils.utils import load_saved_model
from utils.inferer import SlidingWindowInferer
from transforms.call_preproc import call_trans_function

class MONAIInferenceOperator(Operator):
    def __init__(self, config, model_version, output_dir, post_transform, device="cuda", **kwargs):
        super().__init__(**kwargs)
        self.config = config
            
        self.model_path=os.path.join('./models', model_version+'.pth'),
        self.output_dir = output_dir
        self.post_transform = transform
        self.device = device
  
        if "AMP" in config.keys() and config["AMP"]==True:
            self.amp = True
            self.dtype = torch.float16
        else:
            self.amp = False
            self.dtype = torch.float32

        if "DEEP_SUPERVISION" in config.keys() and config["DEEP_SUPERVISION"]==True:
            self.deep_supervision = True
        else: 
            self.deep_supervision = False

        if "ACTIVATION" in config.keys():
            if config["ACTIVATION"].lower()=='sigmoid':
                self.active=torch.nn.Sigmoid()
            elif config["ACTIVATION"].lower()=='softmax':
                self.active=torch.nn.Softmax(dim=0)

        self._load_model()

        self.transforms, _ = call_trans_function(config)
        self.transforms = Compose(self.transforms)

        self.inferer = SlidingWindowInferer(
            roi_size=config["INPUT_SHAPE"],
            sw_batch_size=config["BATCH_SIZE"],
            sw_device=torch.device("cuda"),
            device=torch.device("cpu"),
            overlap=0.5,
            deep_supervision=self.deep_supervision
        )

    def _load_model(self):
        self.model = load_saved_model(self.config, call_model(self.config))

    @torch.no_grad()
    def process(self, input_data):
        """
        input_data: expected to be a dictionary containing the DICOM image volume as a numpy array.
        For example: {"image": image_array, "meta": meta_info}
        """
        # Preprocess image (assuming input_data["image"] is already a numpy array in [C, H, W, D])
        image = self.transforms(input_data["image"])

        # Run inference
        with torch.autocast(enabled=self.amp, dtype=self.dtype, device_type='cuda'):	
            image_tensor = torch.tensor(image, dtype=self.dtype).unsqueeze(0).to(self.device)
            output = self.inferer(image_tensor, network=self.model)[0] 
            output = self.active(output) if not self.active is None else output
        
        torch.cuda.empty_cache()
        return {"output_data": output, "meta": input_data["meta"]}
