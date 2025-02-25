import os
import importlib

from holoscan.core import Application, Graph

from monai_inference_operator import MONAIInferenceOperator
from data_io_operator import (
    ImageLoaderOperator, 
    ImageSaverOperator,
    ResultDisplayOperator
)

def build_pipeline(args):
    # Create Holoscan application
    app = Application(name="MonaiSegmentationApp")

    # Create a graph (pipeline)
    graph = Graph()

    _module = importlib.import_module(f"config.{args.c}")
    config = _module.config
    transform = _module.transform

    # Create other operators (e.g., image reader, display, writer) here...
    # For instance, assume we have an operator 'DICOMReader' that outputs {"image":..., "meta":...}
    # and an operator 'ResultDisplay' that shows the segmentation.
    # These are placeholders for actual implementations.
    image_3d_reader = ImageLoaderOperator(config_name=args.c)
    image_3d_saver = ImageSaverOperator(output_dir=args.o)
    result_display = ResultDisplayOperator(display_interval=1.0)

    # Create your custom MONAI inference operator
    inference_3d_op = MONAIInferenceOperator(
        config=config,
        model_version=config["MODEL_VERSION"],
        output_dir=args.o,
        post_transform=transform,
        name="monai_inference",
    )

    # Connect operators in the graph (3D Segmentation only):
    # image_3d_reader -> inference_3d_op -> result_display, image_3d_saver
    graph.connect(image_3d_reader, "output", inference_3d_op, "input")
    graph.connect(inference_3d_op, "output", image_3d_saver, "input")
    graph.connect(inference_3d_op, "output", result_display, "input")

    # Add graph to application and run:
    app.add_graph(graph)
    return app

if __name__ == "__main__":
	parser = ap.ArgumentParser(description='Segmentation Worker')
	parser.add_argument('-i', type=str, help='Input data path')
	parser.add_argument('-o', type=str, help='Output data path')
	parser.add_argument('-c', type=str, help='Config file name')
	parser.add_argument('-g', type=str, help='GPUs')

	args = parser.parse_args()
    
    app = build_pipeline(args)
    app.run()  # This will run the Holoscan pipeline
