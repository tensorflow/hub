# Copyright 2020 The TensorFlow Hub Authors. All Rights Reserved.
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
"""Utilities to use TF2 SavedModels as feature columns.

Feature columns are compatible with the new FeatureColumn API, see
tensorflow.python.feature_column.feature_column_v2.
"""

import collections

import tensorflow as tf
from tensorflow_hub import keras_layer

# TODO(b/73987364): It is not possible to extend feature columns without
# depending on TensorFlow internal implementation details.
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.feature_column import feature_column_v2
# pylint: enable=g-direct-tensorflow-import


# TODO(b/149367074): Keras can't compute the shape if the input tensor is not
# tf.float32.
def _compute_output_shape(layer, shape, dtype):

  @tf.function(
      input_signature=[tf.TensorSpec(dtype=dtype, name="text", shape=shape)])
  def call(text):
    return layer(text)

  cf = call.get_concrete_function()
  if not isinstance(cf.output_shapes, tf.TensorShape):
    raise ValueError(
        "The SavedModel doesn't return a single result on __call__, "
        "instead it returns %s. Did you specify the right `output_key`?" %
        cf.structured_outputs)
  # Return dimensions after batch size.
  return cf.output_shapes[1:]


def text_embedding_column_v2(key,
                             module_path,
                             output_key=None,
                             trainable=False):
  """Uses a TF2 SavedModel to construct a dense representation from text.

  Args:
    key: A string or `FeatureColumn` identifying the input string data.
    module_path: A string path to the module. Can be a path to local filesystem
      or a tfhub.dev handle.
    output_key: Name of the output item to return if the layer returns a dict.
      If the result is not a single value and an `output_key` is not specified,
      the feature column cannot infer the right output to use.
    trainable: Whether or not the Model is trainable. False by default, meaning
      the pre-trained weights are frozen. This is different from the ordinary
      tf.feature_column.embedding_column(), but that one is intended for
      training from scratch.

  Returns:
    `DenseColumn` that converts from text input.
  """
  if not hasattr(feature_column_v2.StateManager, "has_resource"):
    raise NotImplementedError("The currently used TensorFlow release is not "
                              "compatible. To be compatible, the symbol "
                              "tensorflow.python.feature_column."
                              "feature_column_v2.StateManager.has_resource "
                              "must exist.")

  return _TextEmbeddingColumnV2(
      key=key,
      module_path=module_path,
      output_key=output_key,
      trainable=trainable)


class _TextEmbeddingColumnV2(
    feature_column_v2.DenseColumn,
    collections.namedtuple("_ModuleEmbeddingColumn",
                           ("key", "module_path", "output_key", "trainable"))):
  """Returned by text_embedding_column(). Do not use directly."""

  @property
  def _is_v2_column(self):
    return True

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.key]

  @property
  def _resource_name(self):
    return "hub_text_column_%s" % self.key

  @property
  def name(self):
    """Returns string. Used for variable_scope and naming."""
    if not hasattr(self, "_name"):
      key_name = self.key if isinstance(self.key, str) else self.key.name
      self._name = "{}_hub_module_embedding".format(key_name)
    return self._name

  def create_state(self, state_manager):
    """Imports the module along with all variables."""
    # Note: state_manager._trainable is not public but is the pattern used
    # to propagate the "trainable" state that used to be received via
    # self._get_dense_tensor.
    trainable = self.trainable and state_manager._trainable  # pylint: disable=protected-access
    layer = keras_layer.KerasLayer(
        self.module_path, output_key=self.output_key, trainable=trainable)
    # Note: state manager attaches the loaded resource onto the layer.
    state_manager.add_resource(self, self._resource_name, layer)
    self._variable_shape = _compute_output_shape(layer, [None], tf.string)

  def transform_feature(self, transformation_cache, state_manager):
    return transformation_cache.get(self.key, state_manager)

  @property
  def parse_example_spec(self):
    """Returns a `tf.Example` parsing spec as dict."""
    return {self.key: tf.io.FixedLenFeature([1], tf.string)}

  @property
  def variable_shape(self):
    """`TensorShape` of `get_dense_tensor`, without batch dimension."""
    return self._variable_shape

  def get_dense_tensor(self, transformation_cache, state_manager):
    """Returns a `Tensor`."""
    input_tensor = transformation_cache.get(self, state_manager)
    layer = state_manager.get_resource(self, self._resource_name)
    text_batch = tf.reshape(input_tensor, shape=[-1])
    return layer(text_batch)

  def get_config(self):
    config = dict(zip(self._fields, self))
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None, columns_by_name=None):
    copied_config = config.copy()
    return cls(**copied_config)
