
##!/bin/bash
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

from abc import ABC
from typing import Dict, Optional


class Domain(ABC):
    """Domain Class."""

    def __init__(self, metadata: Optional[Dict] = None):
        """Initialize a Domain object.

        Args:
            metadata (Optional[Dict]): A metadata.
        """
        super().__init__()

        if metadata is not None:
            self._metadata = metadata
        else:
            self._metadata = {}

    def metadata(self) -> Dict:
        return self._metadata
        

from typing import Any, Dict, Optional, Union

try:
    from numpy.typing import _ArrayLike  # type: ignore

    ArrayLike: Any = _ArrayLike
except ImportError:
    ArrayLike = Any

class Image(Domain):
    """_summary_

    Args:
        Domain (_type_): _description_
    """

    def __init__(self, data: Union[ArrayLike], metadata: Optional[Dict] = None):
        """This class encapsulates array-lile object along with its associated metadata dictionary.

        It is designed to represent an image object, without constraining the specific format of its data. Derived
        classes should be created for more specific type of images.
        The default implementation assumes the internal data object is a ndarray, and the metadata dictionary
        is expected to hold the key attributes associated with the original image, e.g., for image converted
        from DICOM instances, the pixel spacings, image orientation patient, etc.

        Args:
            data (Union[ArrayLike]): Numpy Array object.
            metadata (Optional[Dict], optional): A dictionary of the object's metadata. Defaults to None.
        """
        super().__init__(metadata)
        self._data = data

    def asnumpy(self) -> ArrayLike:
        return self._data
