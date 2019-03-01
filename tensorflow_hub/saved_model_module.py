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
"""Module implementation that loads raw SavedModels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_hub import native_module
from tensorflow_hub import saved_model_lib
from tensorflow_hub import tf_v1


_ALWAYS_DROPPED_COLLECTIONS = [
    # SavedModels exported from estimator framework typically contain a
    # collection with the variable that holds the global training step.
    #
    # This collection is ignored when loading it as a module. However the
    # variable that contains the step would still be brought in if declared
    # in the VARIABLES collection.
    tf_v1.GraphKeys.GLOBAL_STEP,

    # SavedModels exported for serving use cases contain collections which refer
    # to ops in the graph that when run are responsible to initialize the
    # session for subsequent signature executions.
    #
    # This generic initialization definition is impossible to support for many
    # hub use cases and therefore the assumption here is that the SavedModel
    # init op can be ignored in favor of initializing using the
    # tf.train.MonitoredSession mechanisms + construction of a new tf.Saver()
    # from the global variables collection.
    tf_v1.saved_model.constants.LEGACY_INIT_OP_KEY,
    tf_v1.saved_model.constants.MAIN_OP_KEY,
]


def _drop_collections(saved_model_handler, collections):
  for meta_graph in saved_model_handler.meta_graphs:
    for collection in collections:
      if collection in meta_graph.collection_def:
        del meta_graph.collection_def[collection]


def create_module_spec_from_saved_model(saved_model_path,
                                        drop_collections=None):
  """Experimental: Create a ModuleSpec out of a SavedModel.

  Define a ModuleSpec from a SavedModel. Note that this is not guaranteed to
  work in all cases and it assumes the SavedModel has followed some conventions:

  - The serialized SaverDef can be ignored and instead can be reconstructed.
  - The init op and main op can be ignored and instead the module can be
    initialized by using the conventions followed by
    `tf.train.MonitoredSession`.

  Note that the set of features supported can increase over time and have side
  effects that were not previously visible. The pattern followed to avoid
  surprises is forcing users to declare which features to ignore (even
  if they are not supported).

  Note that this function creates a ModuleSpec that when exported exports a
  Module (based on a modified copy of the original SavedModel) and not a
  SavedModel.

  Args:
    saved_model_path: Directory with the SavedModel to use.
    drop_collections: Additionally list of collection to drop.

  Returns:
    A ModuleSpec.
  """
  saved_model_handler = saved_model_lib.load(saved_model_path)
  checkpoint_filename = saved_model_lib.get_variables_path(saved_model_path)

  drop_collections = (set(_ALWAYS_DROPPED_COLLECTIONS) |
                      (set(drop_collections) if drop_collections else set()))
  _drop_collections(saved_model_handler, drop_collections)

  return native_module._ModuleSpec(saved_model_handler, checkpoint_filename)  # pylint: disable=protected-access
