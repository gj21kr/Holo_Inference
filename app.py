#!/usr/bin/env python3

import os
import logging
import argparse
import importlib
from pathlib import Path

# from utils.setup_paths import setup_paths
# setup_paths()

from holoscan.core import Application
from holoscan.conditions import CountCondition

from operators.monai_inference_operator import MONAIInferenceOperator
from operators.data_io_operator import (
    ImageLoaderOperator, 
    ImageSaverOperator
)
# from operators.volume_rendering_operator import IntegratedVolumeRendererOp

class MonaiSegmentationApp(Application):
    def __init__(self, *args, **kwargs):
        """Creates an application instance."""
        super().__init__(*args, **kwargs)
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

    def run(self, *args, **kwargs):
        """Run the application."""
        self._logger.info(f"Begin {self.run.__name__}")
        super().run(*args, **kwargs)
        self._logger.info(f"End {self.run.__name__}")

    def compose(self):
        """Creates the app specific operators and chain them up in the processing DAG."""
        self._logger.debug(f"Begin {self.compose.__name__}")
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Segmentation Worker')
        parser.add_argument('-i', type=str, help='Input data path')
        parser.add_argument('-o', type=str, help='Output data path')
        parser.add_argument('-c', type=str, help='Config file name')
        
        args, unknown = parser.parse_known_args(self.argv)
        
        # Set up paths
        input_path = Path(args.i) if args.i else Path("./input")
        output_path = Path(args.o) if args.o else Path("./output")
        config_name = args.c if args.c else "default_config"
        
        self._logger.info(f"Input path: {input_path}, Output path: {output_path}, Config: {config_name}")
        
        # Import configuration
        try:
            _module = importlib.import_module(f"config.{config_name}")
            config = _module.config
            transform = _module.transform
        except ImportError:
            self._logger.error(f"Could not import config module: config.{config_name}")
            raise
        
        # Create operators
        image_loader_op = ImageLoaderOperator(
            self, 
            config=config,
            input_path=input_path,
            name="image_loader_op"
        )
        
        inference_op = MONAIInferenceOperator(
            self,
            config=config,
            model_version=config.get("MODEL_VERSION", "default"),
            output_dir=output_path,
            post_transforms=transform,
            _inferer='sliding_window',
            name="monai_inference_op"
        )

        image_saver_op = ImageSaverOperator(
            self,
            output_dir=output_path,
            name="image_saver_op"
        )

        # holoviz_op = IntegratedVolumeRendererOp(
        #     self,
        #     render_config_file=None,
        #     render_preset_files=None,
        #     density_min=None,
        #     density_max=None,
        #     alloc_width=1024,
        #     alloc_height=768,
        #     window_title="Volume Rendering with ClaraViz",
        #     name="holoviz_op"
        # )
        
        # Connect operators in the flow
        self.add_flow(image_loader_op, inference_op, {(image_loader_op.output_name, inference_op.input_name)})
        self.add_flow(inference_op, image_saver_op, {(inference_op.output_name, image_saver_op.input_name)})
        # self.add_flow(
        #     inference_op,
        #     holoviz_op,
        #     {
        #         (image_loader_op.output_name, "density_volume"),
        #         (image_loader_op.spacing, "density_spacing"),
        #     },
        # )
        # self.add_flow(
        #     inference_op,
        #     holoviz_op,
        #     {
        #         (image_loader_op.output_name, "mask_volume"),
        #         (image_loader_op.spacing, "mask_spacing"),
        #     },
        # )

        self._logger.debug(f"End {self.compose.__name__}")


if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Begin {__name__}")
    
    # Create and run the application
    MonaiSegmentationApp().run()
    
    logging.info(f"End {__name__}")