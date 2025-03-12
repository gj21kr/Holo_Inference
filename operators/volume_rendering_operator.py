# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import logging
import os
import pathlib

from holoscan.core import Operator, OperatorSpec
from holoscan.resources import CudaStreamPool, UnboundedAllocator

# Import operators from holohub
import sys
sys.path.append("/opt/holohub/build/volume_renderer/python/lib")
from holohub.volume_renderer import VolumeRendererOp

logger = logging.getLogger("integrated_volume_rendering")


class JsonLoaderOp(Operator):
    """Operator for loading JSON preset files."""
    
    def __init__(
        self,
        fragment,
        *args,
        **kwargs,
    ):
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("json")
        spec.param("file_names")

    def compute(self, op_input, op_output, context):
        output = []
        for file_name in self.file_names:
            with open(file_name) as f:
                output.append(json.load(f))

        op_output.emit(output, "json")


class IntegratedVolumeRendererOp(Operator):
    """
    An integrated volume renderer operator that combines the functionality of
    volume loading, rendering and visualization for inference applications.
    
    This operator is designed to be used in a Holoscan application to visualize
    CT volumes and inference results.
    """
    
    def __init__(
        self,
        fragment,
        *args,
        render_config_file=None,
        render_preset_files=None,
        density_min=None,
        density_max=None,
        alloc_width=1024,
        alloc_height=768,
        window_title="Volume Rendering with ClaraViz",
        **kwargs,
    ):
        """
        Initialize the integrated volume renderer operator.
        
        Parameters:
        -----------
        fragment : ApplicationContext
            The application context.
        render_config_file : str, optional
            Path to the rendering configuration JSON file.
        render_preset_files : list, optional
            List of paths to renderer preset JSON files.
        density_min : int, optional
            Minimum density value. If not provided, calculated from the data.
        density_max : int, optional
            Maximum density value. If not provided, calculated from the data.
        alloc_width : int, default=1024
            Width of allocated buffer.
        alloc_height : int, default=768
            Height of allocated buffer.
        window_title : str, default="Volume Rendering with ClaraViz"
            Title of the visualization window.
        """
        self._render_config_file = render_config_file
        self._render_preset_files = render_preset_files or []
        self._density_min = density_min
        self._density_max = density_max
        self._alloc_width = alloc_width
        self._alloc_height = alloc_height
        self._window_title = window_title
        
        # Store the child operators we'll create
        self._volume_renderer = None
        self._visualizer = None
        self._preset_loader = None
        
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        # Define inputs
        spec.input("density_volume")
        spec.input("density_spacing")
        spec.input("density_permute_axis", condition="optional")
        spec.input("density_flip_axes", condition="optional")
        
        # Optional mask volume inputs
        spec.input("mask_volume", condition="optional")
        spec.input("mask_spacing", condition="optional")
        spec.input("mask_permute_axis", condition="optional")
        spec.input("mask_flip_axes", condition="optional")
        
        # Camera pose input
        spec.input("camera_pose", condition="optional")
        
        # Outputs
        spec.output("color_buffer_out")
        spec.output("camera_pose_output")
        
        # Parameters
        spec.param("render_config_file", kind="runtime")
        spec.param("render_preset_files", kind="runtime")
        spec.param("density_min", kind="runtime")
        spec.param("density_max", kind="runtime")
        spec.param("alloc_width", kind="runtime")
        spec.param("alloc_height", kind="runtime")
        spec.param("window_title", kind="runtime")

    def initialize(self):
        # Create resources
        fragment = self.fragment
        
        # Create allocator
        volume_allocator = UnboundedAllocator(fragment, name=f"{self.name}_allocator")
        
        # Create CUDA stream pool
        cuda_stream_pool = CudaStreamPool(
            fragment,
            name=f"{self.name}_cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )
        
        # Create volume renderer
        volume_renderer_args = {}
        if self._density_min:
            volume_renderer_args["density_min"] = self._density_min
        if self._density_max:
            volume_renderer_args["density_max"] = self._density_max
            
        self._volume_renderer = VolumeRendererOp(
            fragment,
            name=f"{self.name}_renderer",
            config_file=self._render_config_file,
            allocator=volume_allocator,
            alloc_width=self._alloc_width,
            alloc_height=self._alloc_height,
            cuda_stream_pool=cuda_stream_pool,
            **volume_renderer_args,
        )
        
        # Create HolovizOp for visualization
        from holoscan.operators import HolovizOp
        
        self._visualizer = HolovizOp(
            fragment,
            name=f"{self.name}_viz",
            window_title=self._window_title,
            enable_camera_pose_output=True,
            enable_render_UI=True,  # Enable the render UI for additional controls
            enable_mouse_actions=True,  # Ensure mouse interactions are enabled
            cuda_stream_pool=cuda_stream_pool,
        )
        
        # Create preset loader if preset files are provided
        if self._render_preset_files:
            self._preset_loader = JsonLoaderOp(
                fragment,
                file_names=self._render_preset_files,
                name=f"{self.name}_preset_loader",
            )
            
            # Connect preset loader to volume renderer
            from holoscan.core import ConditionType
            
            fragment.add_flow(self._preset_loader, self._volume_renderer, {("json", "merge_settings")})
            # Set the input condition of merge_settings to NONE
            input_port = self._volume_renderer.spec.inputs["merge_settings:0"]
            if not input_port:
                raise RuntimeError("Could not find `merge_settings:0` input")
            input_port.condition(ConditionType.NONE)
        
        # Connect volume renderer to visualizer
        fragment.add_flow(self._volume_renderer, self._visualizer, {("color_buffer_out", "receivers")})
        fragment.add_flow(self._visualizer, self._volume_renderer, {("camera_pose_output", "camera_pose")})

    def compute(self, op_input, op_output, context):
        # Forward density volume data to the volume renderer
        if "density_volume" in op_input:
            self._volume_renderer.receive(op_input.receive("density_volume"), "density_volume")
        
        if "density_spacing" in op_input:
            self._volume_renderer.receive(op_input.receive("density_spacing"), "density_spacing")
            
        if "density_permute_axis" in op_input:
            self._volume_renderer.receive(op_input.receive("density_permute_axis"), "density_permute_axis")
            
        if "density_flip_axes" in op_input:
            self._volume_renderer.receive(op_input.receive("density_flip_axes"), "density_flip_axes")
        
        # Forward mask volume data if provided
        if "mask_volume" in op_input:
            self._volume_renderer.receive(op_input.receive("mask_volume"), "mask_volume")
            
        if "mask_spacing" in op_input:
            self._volume_renderer.receive(op_input.receive("mask_spacing"), "mask_spacing")
            
        if "mask_permute_axis" in op_input:
            self._volume_renderer.receive(op_input.receive("mask_permute_axis"), "mask_permute_axis")
            
        if "mask_flip_axes" in op_input:
            self._volume_renderer.receive(op_input.receive("mask_flip_axes"), "mask_flip_axes")
        
        # Forward camera pose if provided
        if "camera_pose" in op_input:
            self._volume_renderer.receive(op_input.receive("camera_pose"), "camera_pose")
        
        # Execute the preset loader if we have one
        if self._preset_loader:
            self._preset_loader.compute({}, context)
        
        # Execute the volume renderer
        self._volume_renderer.compute({}, context)
        
        # Execute the visualizer
        self._visualizer.compute({}, context)
        
        # Forward outputs
        if "color_buffer_out" in self._volume_renderer.outputs:
            op_output.emit(self._volume_renderer.outputs["color_buffer_out"], "color_buffer_out")
            
        if "camera_pose_output" in self._visualizer.outputs:
            op_output.emit(self._visualizer.outputs["camera_pose_output"], "camera_pose_output")