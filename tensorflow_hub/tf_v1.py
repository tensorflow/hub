# Copyright 2019 The TensorFlow Hub Authors. All Rights Reserved.
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
"""Util to allow tensorflow_hub to be used both in 1.x and 2.x TensorFlow.

Note: this should not be needed once TF 1.13 is the lowest version to support as
that contains the tf.compat.v1 symbol.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# pylint: disable=g-import-not-at-top
# pylint: disable=unused-import
try:
  from tensorflow.compat.v1 import *  # pylint: disable=wildcard-import
  # The previous line also gets us tensorflow.compat.v1.estimator.
  # Be sure not to import from tensorflow_estimator without version selection.
except ImportError:
  from tensorflow import add_to_collection
  from tensorflow import app
  from tensorflow import assign
  from tensorflow import assign_add
  from tensorflow import AttrValue
  from tensorflow import colocate_with
  from tensorflow import constant_initializer
  from tensorflow import convert_to_tensor_or_indexed_slices
  from tensorflow import estimator
  from tensorflow import feature_column
  from tensorflow import FixedLenFeature
  from tensorflow import fixed_size_partitioner
  from tensorflow import gather
  from tensorflow import get_collection
  from tensorflow import get_collection_ref
  from tensorflow import get_default_graph
  from tensorflow import get_variable
  from tensorflow import get_variable_scope
  from tensorflow import gfile
  from tensorflow import global_variables
  from tensorflow import global_variables_initializer
  from tensorflow import initializers
  from tensorflow import Graph
  from tensorflow import GraphKeys
  from tensorflow import layers
  from tensorflow import losses
  from tensorflow import MetaGraphDef
  from tensorflow import name_scope
  from tensorflow import nn
  from tensorflow import placeholder
  from tensorflow import regex_replace
  from tensorflow import reset_default_graph
  from tensorflow import saved_model
  from tensorflow import Session
  from tensorflow import set_random_seed
  from tensorflow import SparseTensor
  from tensorflow import SparseTensorValue
  from tensorflow import sparse_fill_empty_rows
  from tensorflow import sparse_placeholder
  from tensorflow import sparse_reset_shape
  from tensorflow import sparse_split
  from tensorflow import sparse_tensor_to_dense
  from tensorflow import string_to_hash_bucket_fast
  from tensorflow import train
  from tensorflow import trainable_variables
  from tensorflow import tables_initializer
  from tensorflow import variable_scope
  from tensorflow import zeros_initializer
# pylint: enable=g-import-not-at-top
# pylint: enable=unused-import
