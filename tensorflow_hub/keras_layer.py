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
"""A Keras Layer for using TF Hub modules in TF2 format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json

import six
import tensorflow as tf

from tensorflow_hub import module_v2

# ATTENTION: This file uses private imports from TF2.
# __init__ may not import this file if tensorflow is too old.

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import smart_cond
from tensorflow.python.training.tracking import base as tracking_base
from tensorflow.python.util import tf_inspect
# pylint: enable=g-direct-tensorflow-import


class KerasLayer(tf.keras.layers.Layer):
  """Wraps a Hub module (or a similar callable) for TF2 as a Keras Layer.

  This layer wraps a callable object for use as a Keras layer. The callable
  object can be passed directly, or be specified by a Python string with a
  handle that gets passed to `hub.load()`.

  The callable object is expected to follow the conventions detailed below.
  (These are met by TF2-compatible modules loaded from TensorFlow Hub.)

  The callable is invoked with a single positional argument set to one tensor
  or a list of tensors containing the inputs to the layer. If the callable
  accepts a `training` argument, a Python boolean is passed for it. It is True
  if this layer is marked trainable *and* called for training.

  If present, the following attributes of callable are understood to have
  special meanings:
    variables: a list of all tf.Variable objects that the callable depends on.
    trainable_variables: those elements of `variables` that are reported
      as trainable variables of this Keras Layer when the layer is trainable.
    regularization_losses: a list of callables to be added as losses of this
      Keras Layer when the layer is trainable. Each one must accept zero
      arguments and return a scalar tensor.

  Note: to work-around missing shape inference functionalities from functions
  created from FunctionDefs, in many cases one has to pass an 'output_shape'
  and potentially 'input_shape' and 'dtype'. E.g. the following is a typical
  work-around:
  ```
  hub.KerasLayer(
      "/tmp/text_embedding_model",
      output_shape=[20],  # Outputs a tensor with shape [batch_size, 20].
      input_shape=[],     # Expects a tensor of shape [batch_size] as input.
      dtype=tf.string)    # Expects a tf.string input tensor.
  ```

  Note: This layer can be used inside the model_fn of a TF2 Estimator. See
  https://www.tensorflow.org/alpha/guide/migration_guide#using_a_custom_model_fn
  for guidance on how to pick up trainable variables, losses and updates
  explicitly from Keras objects instead of relying on graph collections.
  This layer class does not support graph collections.

  Args:
    handle: a callable object (subject to the conventions above), or a
      Python string for which hub.load() returns such a callable.
      A string is required to save the Keras config of this Layer.
    trainable: Boolean controlling whether this layer is trainable.
    arguments: optionally, a dict with additional keyword arguments passed
      to the callable. These must be JSON-serializable to save the Keras config
      of this layer.
    **kwargs: 'output_shape': A tuple with the (possibly partial) output
      shape of the callable *without* leading batch size. Other arguments
      are pass into the Layer constructor.
  """

  def __init__(self, handle, trainable=False, arguments=None, **kwargs):
    # Note: for compatibility with keras-model serialization this layer is
    # json-serializable. If you add or change arguments here, please also update
    # the `get_config` method.
    self._handle = handle

    # Resolve the handle to a callable `func`.
    if callable(handle):
      self._func = handle
    else:
      self._func = module_v2.load(handle)
      if not callable(self._func):
        raise ValueError("Non-callable result from hub.load('%s')" %
                         str(handle))
    # TODO(b/124219898): We should do shape inference on the callable.
    if "output_shape" in kwargs:
      self._output_shape = tuple(kwargs.pop("output_shape"))

    # Initialize an empty layer, then add_weight() etc. as needed.
    super(KerasLayer, self).__init__(trainable=trainable, **kwargs)

    # Add trainable and non-trainable weights from the callable.
    if hasattr(self._func, "trainable_variables"):
      for v in self._func.trainable_variables:
        self._add_existing_weight(v, trainable=True)
      trainable_variables = set(self._func.trainable_variables)
    else:
      trainable_variables = set()
    if hasattr(self._func, "variables"):
      for v in self._func.variables:
        if v not in trainable_variables:
          self._add_existing_weight(v, trainable=False)

    # Forward the callable's regularization losses (if any).
    if hasattr(self._func, "regularization_losses"):
      for l in self._func.regularization_losses:
        if not callable(l):
          raise ValueError(
              "hub.KerasLayer(obj) expects obj.regularization_losses to be an "
              "iterable of callables, each returning a scalar loss term.")
        self.add_loss(self._call_loss_if_trainable(l))  # Supports callables.

    # Prepare to call `func`.
    self._func_fullargspec = tf_inspect.getfullargspec(self._func.__call__)
    self._func_wants_training = (
        "training" in self._func_fullargspec.args or
        "training" in self._func_fullargspec.kwonlyargs)
    if arguments is not None:
      self._arguments = arguments

  def _add_existing_weight(self, weight, trainable=None):
    """Calls add_weight() to register but not create an existing weight."""
    if trainable is None: trainable = weight.trainable
    self.add_weight(name=weight.name, shape=weight.shape, dtype=weight.dtype,
                    trainable=trainable, getter=lambda *_, **__: weight)

  def _call_loss_if_trainable(self, loss):
    """Returns `loss` conditioned on whether this layer is trainable."""
    return lambda: loss() if self.trainable else None

  def call(self, inputs, training=None):
    # We basically want to call this...
    kwargs = getattr(self, "_arguments", None)
    if kwargs is None:
      kwargs = {}
    f = functools.partial(self._func, inputs, **kwargs)
    # ...but we may also have to pass a Python boolean for `training`, which
    # is the logical "and" of this layer's trainability and what the surrounding
    # model is doing (analogous to tf.keras.layers.BatchNormalization in TF2).
    # For the latter, we have to look in two places: the `training` argument,
    # or else Keras' global `learning_phase`, which might actually be a tensor.
    if not self._func_wants_training:
      result = f()
    else:
      if self.trainable:
        if training is None:
          training = tf.keras.backend.learning_phase()
      else:
        training = False
      result = smart_cond.smart_cond(training,
                                     lambda: f(training=True),
                                     lambda: f(training=False))
    # TODO(b/124219898): Polymorphic function should return shaped tensor.
    if hasattr(self, "_output_shape"):
      result.set_shape((inputs.shape[0],) + self._output_shape)
    return result

  def get_config(self):
    config = super(KerasLayer, self).get_config()
    if not isinstance(self._handle, six.string_types):
      # Need to raise this type in order for tf.saved_model.save() to fall back
      # to not using config, instead of crashing.
      # TODO(b/134528831): Reconsider the usability implications.
      raise NotImplementedError(
          "Can only generate a valid config for `hub.KerasLayer(handle, ...)`"
          "that uses a string `handle`.\n\n"
          "Got `type(handle)`: {}".format(type(self._handle)))

    config.update({
        "handle": self._handle,
    })

    if hasattr(self, "_output_shape"):
      config["output_shape"] = self._output_shape

    if hasattr(self, "_arguments"):
      # Raise clear errors for non-serializable arguments.
      for key, value in self._arguments.items():
        try:
          json.dumps(value)
        except TypeError as e:
          raise ValueError(
              "`hub.KerasLayer(..., arguments)` contains non json-serializable"
              "values in key: {}".format(key))
      config["arguments"] = self._arguments

    return config
