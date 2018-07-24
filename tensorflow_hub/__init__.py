# Copyright 2018 The TensorFlow Hub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TensorFlow Hub Library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_hub.estimator import LatestModuleExporter
from tensorflow_hub.estimator import register_module_for_export
from tensorflow_hub.feature_column import image_embedding_column
from tensorflow_hub.feature_column import text_embedding_column
from tensorflow_hub.image_util import get_expected_image_size
from tensorflow_hub.image_util import get_num_image_channels
from tensorflow_hub.module import Module
from tensorflow_hub.module_spec import ModuleSpec
from tensorflow_hub.native_module import add_signature
from tensorflow_hub.native_module import create_module_spec
from tensorflow_hub.native_module import load_module_spec
from tensorflow_hub.version import __version__


# Used by doc generation script.
_allowed_symbols = [
    "LatestModuleExporter",
    "register_module_for_export",
    "image_embedding_column",
    "text_embedding_column",
    "get_expected_image_size",
    "get_num_image_channels",
    "Module",
    "ModuleSpec",
    "add_signature",
    "create_module_spec",
    "load_module_spec",
]
