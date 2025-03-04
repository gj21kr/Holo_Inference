# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from abc import ABC

import logging
from os.path import abspath
from typing import Dict, List, Optional, Tuple

from .models.factory import ModelFactory
from operator.utils.model import Model

import argparse
import json
import logging.config
from pathlib import Path
from typing import List, Optional, Union

from operators.medical_imaging.utils import argparse_types

LOG_CONFIG_FILENAME = "logging.json"



class RuntimeEnv(ABC):
    """Class responsible for managing run time settings.

    The expected environment variables are the keys in the defaults dictionary,
    and they can be set to override the defaults.
    """

    ENV_DEFAULT: Dict[str, Tuple[str, ...]] = {
        "input": ("HOLOSCAN_INPUT_PATH", "input"),
        "output": ("HOLOSCAN_OUTPUT_PATH", "output"),
        "model": ("HOLOSCAN_MODEL_PATH", "models"),
        "workdir": ("HOLOSCAN_WORKDIR", ""),
    }

    input: str = ""
    output: str = ""
    model: str = ""
    workdir: str = ""

    def __init__(self, defaults: Optional[Dict[str, Tuple[str, ...]]] = None):
        if defaults is None:
            defaults = self.ENV_DEFAULT
        for key, (env, default) in defaults.items():
            self.__dict__[key] = os.environ.get(env, default)

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses the arguments passed to the application.

    Args:
        argv (Optional[List[str]], optional): The command line arguments to parse.
            The first item should be the path to the python executable.
            If not specified, ``sys.argv`` is used. Defaults to None.

    Returns:
        argparse.Namespace: parsed arguments.
    """
    if argv is None:
        import sys

        argv = sys.argv
    argv = list(argv)  # copy argv for manipulation to avoid side-effects

    # We have intentionally not set the default using `default="INFO"` here so that the default
    # value from here doesn't override the value in `LOG_CONFIG_FILENAME` unless the user indends to do
    # so. If the user doesn't use this flag to set log level, this argument is set to "None"
    # and the logging level specified in `LOG_CONFIG_FILENAME` is used.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-l",
        "--log-level",
        dest="log_level",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=argparse_types.valid_existing_path,
        help="Path to input folder/file (default: input)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=argparse_types.valid_dir_path,
        help="Path to output folder (default: output)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=argparse_types.valid_existing_path,
        help="Path to model(s) folder/file (default: models)",
    )
    parser.add_argument(
        "--workdir",
        "-w",
        type=argparse_types.valid_dir_path,
        help="Path to workspace folder (default: A temporary '.monai_workdir' folder in the current folder)",
    )

    args = parser.parse_args(argv[1:])
    args.argv = argv  # save argv for later use in runpy

    return args


def set_up_logging(level: Optional[str], config_path: Union[str, Path] = LOG_CONFIG_FILENAME):
    """Initializes the logger and sets up logging level.

    Args:
        level (str): A logging level (DEBUG, INFO, WARN, ERROR, CRITICAL).
        log_config_path (str): A path to logging config file.
    """
    # Default log config path
    log_config_path = Path(__file__).absolute().parent.parent / LOG_CONFIG_FILENAME

    config_path = Path(config_path)

    # If a logging config file that is specified by `log_config_path` exists in the current folder,
    # it overrides the default one
    if config_path.exists():
        log_config_path = config_path

    config_dict = json.loads(log_config_path.read_bytes())

    if level is not None and "root" in config_dict:
        config_dict["root"]["level"] = level
    logging.config.dictConfig(config_dict)

class AppContext(object):
    """A class to store the context of an application."""

    def __init__(self, args: Dict[str, str], runtime_env: Optional[RuntimeEnv] = None):
        # Set the args
        self.args: Dict[str, str] = {}
        # Set the runtime environment
        self.runtime_env = runtime_env or RuntimeEnv()

        self._model_loaded = False  # If it has tried to load the models.
        self.model_path = ""  # To be set next.
        self.update(args)

    def update(self, args: Dict[str, str]):
        """Update the context with new args and runtime_env."""
        # Update args
        self.args.update(args)

        # Set the path to input/output/model
        self.input_path = args.get("input") or self.args.get("input") or self.runtime_env.input
        self.output_path = args.get("output") or self.args.get("output") or self.runtime_env.output
        self.workdir = args.get("workdir") or self.args.get("workdir") or self.runtime_env.workdir

        # If model has not been loaded, or the model path has changed, get the path and load model(s)
        old_model_path = self.model_path
        self.model_path = args.get("model") or self.args.get("model") or self.runtime_env.model
        if old_model_path != self.model_path:
            self._model_loaded = False  # path changed, reset the flag to re-load

        if not self._model_loaded:
            self.models: Optional[Model] = ModelFactory.create(abspath(self.model_path))
            self._model_loaded = True

    def __repr__(self):
        return (
            f"AppContext(input_path={self.input_path}, output_path={self.output_path}, "
            f"model_path={self.model_path}, workdir={self.workdir})"
        )


def init_app_context(
    argv: Optional[List[str]] = None, runtime_env: Optional[RuntimeEnv] = None
) -> AppContext:
    """Initializes the app context with arguments and well-known environment variables.

    The arguments, if passed in, override the attributes set with environment variables.

    Args:
        argv (Optional[List[str]], optional): arguments passed to the program. Defaults to None.

    Returns:
        AppContext: the AppContext object
    """

    args = parse_args(argv)
    set_up_logging(args.log_level)
    logging.info(f"Parsed args: {args}")

    # The parsed args from the command line override that from the environment variables
    app_context = AppContext({key: val for key, val in vars(args).items() if val}, runtime_env)
    logging.info(f"AppContext object: {app_context}")

    return app_context