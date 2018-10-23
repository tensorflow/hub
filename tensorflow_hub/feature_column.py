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
"""Utilities to use Modules as feature columns."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf
from tensorflow_hub import image_util
from tensorflow_hub import module

# TODO(b/73987364): It is not possible to extend feature columns without
# depending on TensorFlow internal implementation details.
from tensorflow.python.feature_column import feature_column


def text_embedding_column(key, module_spec, trainable=False):
  """Uses a Module to construct a dense representation from a text feature.

  This feature column can be used on an input feature whose values are strings
  of arbitrary size.

  The result of this feature column is the result of passing its `input`
  through the module `m` instantiated from `module_spec`, as per
  `result = m(input)`. The `result` must have dtype float32 and shape
  `[batch_size, num_features]` with a known value of num_features.

  Example:

  ```python
    comment = text_embedding_column("comment", "/tmp/text-module")
    feature_columns = [comment, ...]
    ...
    features = {
      "comment": np.array(["wow, much amazing", "so easy", ...]),
      ...
    }
    labels = np.array([[1], [0], ...])
    input_fn = tf.estimator.inputs.numpy_input_fn(features, labels, shuffle=True)
    estimator = tf.estimator.DNNClassifier(hidden_units, feature_columns)
    estimator.train(input_fn, max_steps=100)
  ```

  Args:
    key: A string or `_FeatureColumn` identifying the text feature.
    module_spec: A ModuleSpec defining the Module to instantiate or a path where
      to load a ModuleSpec via `load_module_spec`
    trainable: Whether or not the Module is trainable. False by default,
      meaning the pre-trained weights are frozen. This is different from the
      ordinary tf.feature_column.embedding_column(), but that one is intended
      for training from scratch.

  Returns:
    `_DenseColumn` that converts from text input.

  Raises:
     ValueError: if module_spec is not suitable for use in this feature column.
  """
  module_spec = module.as_module_spec(module_spec)
  _check_module_is_text_embedding(module_spec)
  return _TextEmbeddingColumn(key=key, module_spec=module_spec,
                              trainable=trainable)


def _check_module_is_text_embedding(module_spec):
  """Raises ValueError if `module_spec` is not a text-embedding module.

  Args:
    module_spec: A `ModuleSpec` to test.

  Raises:
    ValueError: if `module_spec` default signature is not compatible with
    Tensor(string, shape=(?,)) -> Tensor(float32, shape=(?,K)).
  """
  issues = []

  # Find issues with signature inputs.
  input_info_dict = module_spec.get_input_info_dict()
  if len(input_info_dict) != 1:
    issues.append("Module default signature must require only one input")
  else:
    input_info, = input_info_dict.values()
    input_shape = input_info.get_shape()
    if not (input_info.dtype == tf.string and input_shape.ndims == 1 and
            input_shape.as_list() == [None]):
      issues.append(
          "Module default signature must have only one input "
          "tf.Tensor(shape=(?,), dtype=string)"
      )

  # Find issues with signature outputs.
  output_info_dict = module_spec.get_output_info_dict()
  if "default" not in output_info_dict:
    issues.append("Module default signature must have a 'default' output.")
  else:
    output_info = output_info_dict["default"]
    output_shape = output_info.get_shape()
    if not (output_info.dtype == tf.float32 and output_shape.ndims == 2 and
            not output_shape.as_list()[0] and output_shape.as_list()[1]):
      issues.append(
          "Module default signature must have a 'default' output of "
          "tf.Tensor(shape=(?,K), dtype=float32)."
      )

  if issues:
    raise ValueError("Module is not a text-embedding: %r" % issues)


class _TextEmbeddingColumn(
    feature_column._DenseColumn,  # pylint: disable=protected-access
    collections.namedtuple("_ModuleEmbeddingColumn",
                           ("key", "module_spec", "trainable"))):
  """Returned by text_embedding_column(). Do not use directly."""

  @property
  def name(self):
    """Returns string. Used for variable_scope and naming."""
    if not hasattr(self, "_name"):
      self._name = "{}_hub_module_embedding".format(self.key)
    return self._name

  def _transform_feature(self, inputs):
    """Returns intermediate representation (usually a `Tensor`)."""
    return inputs.get(self.key)

  @property
  def _parse_example_spec(self):
    """Returns a `tf.Example` parsing spec as dict."""
    return {self.key: tf.FixedLenFeature([1], tf.string)}

  @property
  def _variable_shape(self):
    """`TensorShape` of `_get_dense_tensor`, without batch dimension."""
    return self.module_spec.get_output_info_dict()["default"].get_shape()[1:]

  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    """Returns a `Tensor`."""
    del weight_collections
    text_batch = tf.reshape(inputs.get(self), shape=[-1])
    m = module.Module(self.module_spec, trainable=self.trainable and trainable)
    return m(text_batch)


def image_embedding_column(key, module_spec):
  """Uses a Module to get a dense 1-D representation from the pixels of images.

  This feature column can be used on images, represented as float32 tensors of
  RGB pixel data in the range [0,1]. This can be read from a numeric_column()
  if the tf.Example input data happens to have decoded images, all with the
  same shape [height, width, 3]. More commonly, the input_fn will have code to
  explicitly decode images, resize them (possibly after performing data
  augmentation such as random crops etc.), and provide a batch of shape
  [batch_size, height, width, 3].

  The result of this feature column is the result of passing its `input`
  through the module `m` instantiated from `module_spec`, as per
  `result = m({"images": input})`. The `result` must have dtype float32 and
  shape `[batch_size, num_features]` with a known value of num_features.

  Example:

  ```python
    image_column = hub.image_embedding_column("embeddings", "/tmp/image-module")
    feature_columns = [image_column, ...]
    estimator = tf.estimator.LinearClassifier(feature_columns, ...)
    height, width = hub.get_expected_image_size(image_column.module_spec)
    input_fn = ...  # Provides "embeddings" with shape [None, height, width, 3].
    estimator.train(input_fn, ...)
  ```

  Args:
    key: A string or `_FeatureColumn` identifying the input image data.
    module_spec: A string handle or a `ModuleSpec` identifying the module.

  Returns:
    `_DenseColumn` that converts from pixel data.

  Raises:
     ValueError: if module_spec is not suitable for use in this feature column.
  """
  module_spec = module.as_module_spec(module_spec)
  _check_module_is_image_embedding(module_spec)
  return _ImageEmbeddingColumn(key=key, module_spec=module_spec)


def _check_module_is_image_embedding(module_spec):
  """Raises ValueError if `module_spec` is not usable as image embedding.

  Args:
    module_spec: A `_ModuleSpec` to test.

  Raises:
    ValueError: if `module_spec` default signature is not compatible with
        mappingan "images" input to a Tensor(float32, shape=(_,K)).
  """
  issues = []

  # Find issues with "default" signature inputs. The common signatures for
  # image models prescribe a specific name; we trust it if we find it
  # and if we can do the necessary inference of input shapes from it.
  input_info_dict = module_spec.get_input_info_dict()
  if (list(input_info_dict.keys()) != ["images"] or
      input_info_dict["images"].dtype != tf.float32):
    issues.append("Module 'default' signature must require a single input, "
                  "which must have type float32 and name 'images'.")
  else:
    try:
      image_util.get_expected_image_size(module_spec)
    except ValueError as e:
      issues.append("Module does not support hub.get_expected_image_size(); "
                    "original error was:\n" + str(e))  # Raised again below.

  # Find issues with "default" signature outputs. We test that the dtype and
  # shape is appropriate for use in input_layer().
  output_info_dict = module_spec.get_output_info_dict()
  if "default" not in output_info_dict:
    issues.append("Module 'default' signature must have a 'default' output.")
  else:
    output_type = output_info_dict["default"].dtype
    output_shape = output_info_dict["default"].get_shape()
    if not (output_type == tf.float32 and output_shape.ndims == 2 and
            output_shape.dims[1].value):
      issues.append("Module 'default' signature must have a 'default' output "
                    "of tf.Tensor(shape=(_,K), dtype=float32).")

  if issues:
    raise ValueError("Module is not usable as image embedding: %r" % issues)


class _ImageEmbeddingColumn(
    feature_column._DenseColumn,  # pylint: disable=protected-access
    collections.namedtuple("_ImageEmbeddingColumn", ("key", "module_spec"))):
  """Returned by image_embedding_column(). Do not use directly."""

  @property
  def name(self):
    """Returns string. Used for variable_scope and naming."""
    if not hasattr(self, "_name"):
      self._name = "{}_hub_module_embedding".format(self.key)
    return self._name

  def _transform_feature(self, inputs):
    """Returns intermediate representation (usually a `Tensor`)."""
    return inputs.get(self.key)

  @property
  def _parse_example_spec(self):
    """Returns a `tf.Example` parsing spec as dict."""
    height, width = image_util.get_expected_image_size(self.module_spec)
    input_shape = [height, width, 3]
    return {self.key: tf.FixedLenFeature(input_shape, tf.float32)}

  @property
  def _variable_shape(self):
    """`TensorShape` of `_get_dense_tensor`, without batch dimension."""
    return self.module_spec.get_output_info_dict()["default"].get_shape()[1:]

  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    """Returns a `Tensor` to represent this feature in the input_layer()."""
    del weight_collections, trainable  # Unused.
    m = module.Module(self.module_spec, trainable=False)
    images = inputs.get(self)
    return m({"images": images})
