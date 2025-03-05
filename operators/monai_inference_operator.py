import os
from typing import Dict, Sequence, Union
import logging
from collections.abc import Hashable, Mapping
from threading import Lock

import torch
import nibabel as nib
import numpy as np
from numpy import uint8

# Import Holoscan base operator (the exact import may vary based on Holoscan version)
from holoscan.core import ConditionType, Fragment, Operator, OperatorSpec

# Third-party
from core.call import call_model
from utils.utils import load_saved_model
from transforms.call_preproc import call_trans_function
from transforms.ImageProcessing import Antialiasingd   
from operators.utils.monai_inferer_operator import InfererType, InMemImageReader 

# MONAI imports
import monai
from monai.transforms import (
    Compose, Activationsd,
    AsDiscreted, SaveImaged, MapTransform,
)
from monai.data import decollate_batch
from monai.data import Dataset, DataLoader
from monai.data import ImageReader as ImageReader_
from monai.inferers import sliding_window_inference
from monai.inferers import SimpleInferer as simple_inference
from monai.transforms import Compose
from monai.utils import MetaKeys, SpaceKeys, optional_import, ensure_tuple


class MONAIInferenceOperator(Operator):
    def __init__(
        self,
        fragment: Fragment, *args,
        config, model_version, output_dir, post_transforms, _inferer, **kwargs):
        
        self.input_name = 'image'
        self.output_name = 'prediction'

        self._input_dataset_key = "image"
        self._pred_dataset_key = "pred"

        self._inferer = InfererType.SLIDING_WINDOW if _inferer=='sliding_window' else InfererType.SIMPLE

        self._lock = Lock()
        self._executing = False

        self.config = config
        self._roi_size = config.get("INPUT_SHAPE", (96, 96, 96))
        self._overlap = 0.25
        self._sw_batch_size = 12
                    
        self.model_path=os.path.join('./models', model_version+'.pth'),
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
                    output_dir=self.output_dir,
                    output_postfix="seg",
                    output_dtype=uint8,
                    resample=False,
                    output_ext=".nii",
                ),
            ]
            self.post_process = Compose(self.post_transforms)

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name)
        spec.output(self.output_name)

    def _load_model(self):
        self.model = load_saved_model(self.config, call_model(self.config))


    @property
    def roi_size(self):
        """The ROI size of tensors used in prediction."""
        return self._roi_size

    @roi_size.setter
    def roi_size(self, roi_size: Union[Sequence[int], int]):
        self._roi_size = ensure_tuple(roi_size)

    @property
    def input_dataset_key(self):
        """This is the input image key name used in dictionary based MONAI pre-transforms."""
        return self._input_dataset_key

    @input_dataset_key.setter
    def input_dataset_key(self, val: str):
        if not val or len(val) < 1:
            raise ValueError("Value cannot be None or blank.")
        self._input_dataset_key = val

    @property
    def pred_dataset_key(self):
        """This is the prediction key name used in dictionary based MONAI post-transforms."""
        return self._pred_dataset_key

    @pred_dataset_key.setter
    def pred_dataset_key(self, val: str):
        if not val or len(val) < 1:
            raise ValueError("Value cannot be None or blank.")
        self._pred_dataset_key = val

    @property
    def overlap(self):
        """This is the overlap used during sliding window inference"""
        return self._overlap

    @overlap.setter
    def overlap(self, val: float):
        if val < 0 or val > 1:
            raise ValueError("Overlap must be between 0 and 1.")
        self._overlap = val

    @property
    def sw_batch_size(self):
        """The batch size to run window slices"""
        return self._sw_batch_size

    @sw_batch_size.setter
    def sw_batch_size(self, val: int):
        if not isinstance(val, int) or val < 0:
            raise ValueError("sw_batch_size must be a positive integer.")
        self._sw_batch_size = val

    @property
    def inferer(self) -> Union[InfererType, str]:
        """The type of inferer to use"""
        return self._inferer

    @inferer.setter
    def inferer(self, val: InfererType):
        if not isinstance(val, InfererType):
            raise ValueError(f"Value must be of the correct type {InfererType}.")
        self._inferer = val

    def compute(self, op_input, op_output, context):
        """Infers with the input image and save the predicted image to output

        Args:
            op_input (InputContext): An input context for the operator.
            op_output (OutputContext): An output context for the operator.
            context (ExecutionContext): An execution context for the operator.
        """

        with self._lock:
            if self._executing:
                raise RuntimeError("Operator is already executing.")
            else:
                self._executing = True
        try:
            input_image = op_input.receive(self.input_name)
            if input_image is None:
                raise ValueError("Input is None.")
            op_output.emit(self.compute_impl(input_image, context), self.output_name)
        finally:
            # Reset state on completing this method execution.
            with self._lock:
                self._executing = False

    def compute_impl(self, input_data, context):
        if input_data is None:
            raise ValueError("Input is None.")

        input_image = input_data["image"]
        input_img_metadata = input_data["meta"]
        self._reader = InMemImageReader(input_image)

        device = torch.device(self.device)
        dataset = Dataset(data=[{self._input_dataset_key: input_image}], transform=self.pre_process)
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0
        )  # Should the batch_size be dynamic?

        with torch.no_grad():
            with torch.amp.autocast(self.device, enabled=self.amp):
                for d in dataloader:
                    images = d[self._input_dataset_key].to(device)
                    if images.ndim == 3:
                        images = images.unsqueeze(0).unsqueeze(0)
                    elif images.ndim == 4:
                        images = images.unsqueeze(0)

                    if self._inferer == InfererType.SLIDING_WINDOW:
                        d[self._pred_dataset_key] = sliding_window_inference(
                            inputs=images,
                            roi_size=self.roi_size,
                            mode="gaussian",
                            progress=True,
                            sw_batch_size=self.sw_batch_size,
                            overlap=self.overlap,
                            predictor=self.model,
                        )
                        # d[self._pred_dataset_key] = SlidingWindowInfererAdapt(
                        #     roi_size=self.roi_size,
                        #     mode="gaussian",
                        #     progress=True,
                        #     sw_batch_size=self.sw_batch_size,
                        #     overlap=self.overlap,
                        #     predictor=self.model,
                        # )(images)
                    elif self._inferer == InfererType.SIMPLE:
                        # Instantiates the SimpleInferer and directly uses its __call__ function
                        d[self._pred_dataset_key] = simple_inference()(
                            inputs=images, network=self.model
                        )
                    else:
                        raise ValueError(
                            f"Unknown inferer: {self._inferer!r}. Available options are "
                            f"{InfererType.SLIDING_WINDOW!r} and {InfererType.SIMPLE!r}."
                        )

                    d = [self.post_process(i) for i in decollate_batch(d)]
                    out_ndarray = d[0][self._pred_dataset_key].cpu().numpy()
                    # Need to squeeze out the channel dim fist
                    out_ndarray = np.squeeze(out_ndarray, 0)
                    # NOTE: The domain Image object simply contains a Arraylike obj as image as of now.
                    #       When the original DICOM series is converted by the Series to Volume operator,
                    #       using pydicom pixel_array, the 2D ndarray of each slice has index order HW, and
                    #       when all slices are stacked with depth as first axis, DHW. In the pre-transforms,
                    #       the image gets transposed to WHD and used as such in the inference pipeline.
                    #       So once post-transforms have completed, and the channel is squeezed out,
                    #       the resultant ndarray for the prediction image needs to be transposed back, so the
                    #       array index order is back to DHW, the same order as the in-memory input Image obj.
                    out_ndarray = out_ndarray.T.astype(np.uint8)
                    self._logger.info(f"Output Seg image numpy array shaped: {out_ndarray.shape}")
                    self._logger.info(f"Output Seg image pixel max value: {np.amax(out_ndarray)}")

                    return Image(out_ndarray, input_img_metadata)
