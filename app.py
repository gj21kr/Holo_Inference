#!/usr/bin/env python3

import os
import logging
import argparse
import importlib
from pathlib import Path

from holoscan.conditions import CountCondition
from holoscan.core import Application

from monai_inference_operator import MONAIInferenceOperator
from data_io_operator import (
    ImageLoaderOperator, 
    ImageSaverOperator,
    ResultDisplayOperator
)

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
        parser.add_argument('-g', type=str, help='GPUs', default='0')
        
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
            post_transform=transform,
            name="monai_inference_op"
        )

        result_display_op = ResultDisplayOperator(
            self,
            display_interval=1.0,
            name="result_display_op"
        )
        
        # Connect operators in the flow (similar to `add_flow` in paste.txt)
        self.add_flow(image_loader_op, inference_op, {(image_loader_op.output_name, inference_op.input_name)})
        self.add_flow(inference_op, result_display_op, {(inference_op.output_name, "input")})
        
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