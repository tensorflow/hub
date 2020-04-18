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

from absl import logging
from distutils.version import LooseVersion
import tensorflow as tf

# pylint: disable=g-import-not-at-top
# Only do imports after check TensorFlow version so the useful
# error message is thrown instead of an obscure error of missing
# symbols at executing the imports.
from tensorflow_hub.estimator import LatestModuleExporter
from tensorflow_hub.estimator import register_module_for_export
from tensorflow_hub.feature_column import image_embedding_column
from tensorflow_hub.feature_column import sparse_text_embedding_column
from tensorflow_hub.feature_column import text_embedding_column
from tensorflow_hub.feature_column_v2 import text_embedding_column_v2
from tensorflow_hub.image_util import attach_image_module_info
from tensorflow_hub.image_util import get_expected_image_size
from tensorflow_hub.image_util import get_num_image_channels
from tensorflow_hub.image_util import ImageModuleInfo
from tensorflow_hub.module import eval_function_for_module
from tensorflow_hub.module import load_module_spec
from tensorflow_hub.module import Module
from tensorflow_hub.module_v2 import load
from tensorflow_hub.module_v2 import resolve
from tensorflow_hub.module_spec import ModuleSpec
from tensorflow_hub.native_module import add_signature
from tensorflow_hub.native_module import attach_message
from tensorflow_hub.native_module import create_module_spec
from tensorflow_hub.saved_model_module import create_module_spec_from_saved_model
from tensorflow_hub.version import __version__

# pylint: disable=g-bad-import-order
# The following imports may fail if TensorFlow is too old for TF2 features.
try:
  from tensorflow_hub.keras_layer import KerasLayer
except ImportError:
  if LooseVersion(tf.__version__) < LooseVersion("1.14.0"):
    logging.info("hub.KerasLayer is not available "
                 "because TensorFlow version is less than 1.14")
  else:
    raise  # This is unexpected and indicates a problem.

from tensorflow_hub.config import _run
_run()

# The package `tensorflow_hub.tools` is available separately for import, but
# it is not meant to be available as attribute of the tensorflow_hub module.
from tensorflow_hub import tools
del tools
# pylint: enable=g-bad-import-order
# pylint: enable=g-import-not-at-top

# If __all__ is defined the doc generator script only documents the listed
# objects (__all__ defines which symbols you get with
# `from tensorflow_hub import *`).
__all__ = [
    "LatestModuleExporter",
    "register_module_for_export",
    "image_embedding_column",
    "sparse_text_embedding_column",
    "text_embedding_column",
    "text_embedding_column_v2",
    "attach_image_module_info",
    "get_expected_image_size",
    "get_num_image_channels",
    "ImageModuleInfo",
    "KerasLayer",
    "Module",
    "ModuleSpec",
    "add_signature",
    "attach_message",
    "create_module_spec",
    "create_module_spec_from_saved_model",
    "load",
    "load_module_spec",
    "resolve",
]
