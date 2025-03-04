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
from holoscan.core import ConditionType, Fragment, Operator, OperatorSpec

# Third-party
from core.call import call_model
from utils.utils import load_saved_model
from utils.inferer import SlidingWindowInferer
from transforms.call_preproc import call_trans_function

class MONAIInferenceOperator(Operator):
    def __init__(
        self,
        fragment: Fragment, *args,
        config, model_version, output_dir, post_transforms, device="cuda", **kwargs):
        
        self.input_name = 'image'
        self.output_name = 'prediction'

        self._input_dataset_key = "image"
        self._pred_dataset_key = "pred"

        self.config = config
            
        self.model_path=os.path.join('./models', model_version+'.pth'),
        self.output_dir = output_dir
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

        self._load_model()

        self.pre_transforms, _ = call_trans_function(config)
        self.pre_process = Compose(self.pre_transforms)

        if "ACTIVATION" in config.keys():
            if config["ACTIVATION"].lower()=='sigmoid':
                self.post_transforms = [
                    Activationsd(keys=self._pred_dataset_key, sigmoid=True),
                    AsDiscreted(keys=self._pred_dataset_key, argmax=True),
                    ] 
            elif config["ACTIVATION"].lower()=='softmax':
                self.post_transforms = [
                    Activationsd(keys=self._pred_dataset_key, softmax=True),
                    AsDiscreted(keys=self._pred_dataset_key, argmax=True),
                    ] 
                    
            self.post_transforms += post_transforms + [
                # Smoothen segmentation volume
                Antialiasingd(
                    keys=self._pred_dataset_key,
                ),
                # The SaveImaged transform can be commented out to save 5 seconds.
                # Uncompress NIfTI file, nii, is used favoring speed over size, but can be changed to nii.gz
                SaveImaged(
                    keys=self._pred_dataset_key,
                    output_dir=out_dir,
                    output_postfix="seg",
                    output_dtype=uint8,
                    resample=False,
                    output_ext=".nii",
                ),
            ]
            self.post_process = Compose(self.post_transforms)

        self.inferer = SlidingWindowInferer(
            roi_size=config["INPUT_SHAPE"],
            sw_batch_size=config["BATCH_SIZE"],
            sw_device=torch.device("cuda"),
            device=torch.device("cpu"),
            overlap=0.5,
            deep_supervision=self.deep_supervision
        )
        super().__init__(fragment, *args, **kwargs)


    def _load_model(self):
        self.model = load_saved_model(self.config, call_model(self.config))

    @torch.no_grad()
    def process(self, input_data):
        """
        input_data: expected to be a dictionary containing the DICOM image volume as a numpy array.
        For example: {"image": image_array, "meta": meta_info}
        """

        # Run inference
        with torch.autocast(enabled=self.amp, dtype=self.dtype, device_type='cuda'):	
            image_tensor = torch.from_numpy(image).to(self.device)
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
            elif image_tensor.ndim == 4:
                image_tensor = image_tensor.unsqueeze(0)
            # Delegates inference and saving output to the built-in operator.
            infer_operator = MonaiSegInferenceOperator(
                self.fragment,
                roi_size=self.config["INPUT_SHAPE"],
                pre_transforms=self.pre_transforms,
                post_transforms=self.post_transforms,
                overlap=0.5,
                model=self.model,
                inferer=InfererType.SLIDING_WINDOW,
                sw_batch_size=1,
                name="monai_seg_inference_op",
            )

            # Setting the keys used in the dictionary based transforms
            infer_operator.input_dataset_key = self._input_dataset_key
            infer_operator.pred_dataset_key = self._pred_dataset_key

            # Now emit data to the output ports of this operator
            op_output.emit(infer_operator.compute_impl(input_image, context), self.output_name)
            self._logger.debug(
                f"Setting {self.output_name_saved_images_folder} with {self.output_folder}"
            )
