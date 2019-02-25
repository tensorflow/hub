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

import tensorflow as tf

from tensorflow_hub import module

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
  accepts a `training` argument, a Python boolean is passed for it.

  If present, the following attributes of callable are understood to have
  special meanings:
    variables: a list of all tf.Variable objects that the callable depends on.
    trainable_variables: those elements of `variables` that are reported
      as trainable variables of this Keras Layer.
    regularization_losses: a list of callables to be added as losses of this
      Keras Layer. Each one must accept zero arguments and return a scalar
      tensor.

  Args:
    handle: a callable object (subject to the conventions above), or a
      Python string for which hub.load() returns such a callable.
    output_shape: A tuple with the (possibly partial) output shape of the
      callable *without* leading batch size.
    trainable: Boolean controlling whether the trainable variables of the
      callable are reported as trainable variables of this layer.
    arguments: optionally, a dict with additional keyword arguments passed
      to the callable.
  """

  @tracking_base.no_automatic_dependency_tracking
  def __init__(self, handle, output_shape, trainable=False, arguments=None,
               **kwargs):
    # Resolve the handle to a callable `func`.
    if callable(handle):
      self._func = handle
    else:
      self._func = module.load(handle)
      if not callable(self._func):
        raise ValueError("Non-callable result from hub.load('%s')" %
                         str(handle))

    # Set self._{non,}_trainable_weights and then call Layer.__init__.
    # This together with @no_automatic_dependency_tracking above preserves
    # func.trainable_variables independent of tf.Variable(..., trainable=...).
    if hasattr(self._func, "trainable_variables"):
      self._trainable_weights = [v for v in self._func.trainable_variables]
      trainable_variables_set = set(self._func.trainable_variables)
    else:
      self._trainable_weights = []
      trainable_variables_set = set()
    if hasattr(self._func, "variables"):
      self._non_trainable_weights = [v for v in self._func.variables
                                     if v not in trainable_variables_set]
    else:
      self._non_trainable_weights = []
    super(KerasLayer, self).__init__(trainable=trainable, **kwargs)

    # Prepare to call `func`.
    self._func_fullargspec = tf_inspect.getfullargspec(self._func.__call__)
    self._func_wants_training = (
        "training" in self._func_fullargspec.args or
        "training" in self._func_fullargspec.kwonlyargs)
    self._arguments = arguments or {}
    # TODO(b/124219898): We should be able to get the embedding dimension from
    # the restored model.
    self._output_shape = tuple(output_shape)

    # Forward the callable's regularization losses (if any).
    if hasattr(self._func, "regularization_losses"):
      for l in self._func.regularization_losses:
        if not callable(l):
          raise ValueError(
              "hub.KerasLayer(obj) expects obj.regularization_losses to be an "
              "iterable of callables, each returning a scalar loss term.")
        self.add_loss(l)  # Supports callables.

  def call(self, x, training=None):
    # We basically want to call this...
    f = functools.partial(self._func, x, **self._arguments)
    # ...but we may also have to pass a Python boolean for `training`.
    if not self._func_wants_training:
      result = f()
    else:
      if training is None:
        training = tf.keras.backend.learning_phase()  # Could be a tensor.
      result = smart_cond.smart_cond(training,
                                     lambda: f(training=True),
                                     lambda: f(training=False))
    # TODO(b/124219898): Polymorphic function should return shaped tensor.
    result.set_shape(self.compute_output_shape(x.shape))
    return result

  def compute_output_shape(self, input_shape):  # Override this Layer method.
    return (input_shape[0],) + self._output_shape
