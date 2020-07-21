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

from absl import logging
import six
import tensorflow as tf

from tensorflow_hub import module_v2

# ATTENTION: This file uses private imports from TF2.
# __init__ may not import this file if tensorflow is too old.

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import smart_cond
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import tf_inspect
# pylint: enable=g-direct-tensorflow-import


class KerasLayer(tf.keras.layers.Layer):
  """Wraps a SavedModel (or a legacy TF1 Hub format) as a Keras Layer.

  This layer wraps a callable object for use as a Keras layer. The callable
  object can be passed directly, or be specified by a Python string with a
  handle that gets passed to `hub.load()`.

  This is the preferred API to load a TF2-style SavedModel from TF Hub
  into a Keras model. Calling this function requires TF 1.15 or newer.
  It can be called both in eager and graph mode.

  The callable object is expected to follow the conventions detailed below.
  (These are met by TF2-compatible modules loaded from TensorFlow Hub.)

  The callable is invoked with a single positional argument set to one tensor
  or a nest of tensors containing the inputs to the layer. If the callable
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
  created from FunctionDefs, in rare cases one has to pass an 'output_shape'
  and potentially 'input_shape' and 'dtype'. E.g. the following is a typical
  work-around:
  ```
  hub.KerasLayer(
      "/tmp/text_embedding_model",
      output_shape=[20],  # Outputs a tensor with shape [batch_size, 20].
      input_shape=[],     # Expects a tensor of shape [batch_size] as input.
      dtype=tf.string)    # Expects a tf.string input tensor.
  ```

  Note: This layer can be used inside the model_fn of a TF2 Estimator. See the
  [migration guide]
  (https://www.tensorflow.org/beta/guide/migration_guide#using_a_custom_model_fn)
  for guidance on how to pick up trainable variables, losses and updates
  explicitly from Keras objects instead of relying on graph collections.
  This layer class does not support graph collections.
  Distributed training of the Estimator requires setting the option
  `session_config.experimental.share_cluster_devices_in_session` within
  the `tf.estimator.RunConfig`. (It becomes non-experimental in TF2.2.)

  Note: The data types used by a saved model have been fixed at saving time.
  Using tf.keras.mixed_precision etc. has no effect on the saved model
  that gets loaded by a hub.KerasLayer.

  Attributes:
    handle: A callable object (subject to the conventions above), or a Python
      string to load a saved model via hub.load(). A string is required to save
      the Keras config of this Layer.
    trainable: Optional. A boolean controlling whether this layer is trainable.
      Must not be set to True when using a signature (raises ValueError),
      including the use of legacy TF1 Hub format.
    arguments: Optional. A dict with additional keyword arguments passed to the
      callable. These must be JSON-serializable to save the Keras config of this
      layer, and are not tracked as checkpointing dependencies of this layer.
    _sentinel: Used to prevent further positional arguments.
    tags: Optional. If set indicates which graph variant to use. For legacy
      models in TF1 Hub format leaving unset means to use the empty tags set.
    signature: Optional. If set, KerasLayer will use the requested signature.
      For legacy models in TF1 Hub format leaving unset means to use the
      `default` signature. When using a signature, either
      signature_outputs_as_dict or output_key have to set.
    signature_outputs_as_dict: If set to True, the call to this layer returns a
      dict of all the signature outputs. Can only be used if a signature is
      specified (or default signature is used for legacy models in TF1 Hub
      format).
    output_key: Name of the output item to return if the layer returns a dict.
      For legacy models in TF1 Hub format leaving unset means to return the
      `default` output.
    output_shape: A tuple or a nest of tuples with the (possibly partial) output
      shapes of the callable *without* leading batch size. This must have the
      same nesting structure as the output of the callable object and cover all
      output tensors.
    load_options: Optional, `tf.saved_model.LoadOptions` object that specifies
      options for loading when a Python string is provided as `handle`. This
      argument can only be used from TensorFlow 2.3 onwards.
    **kwargs: Forwarded to Keras' base Layer constructor.
  """

  def __init__(
      self,
      handle,
      trainable=False,
      arguments=None,
      _sentinel=None,  # pylint: disable=invalid-name
      tags=None,
      signature=None,
      signature_outputs_as_dict=None,
      output_key=None,
      output_shape=None,
      load_options=None,
      **kwargs):
    # Note: for compatibility with keras-model serialization this layer is
    # json-serializable. If you add or change arguments here, please also update
    # the `get_config` method.
    # The arguments are marked NoDependency to avoid autoconversion to a
    # trackable _DictWrapper, because that upsets json.dumps() when saving
    # the result of get_config().
    self._handle = handle
    self._arguments = data_structures.NoDependency(arguments or {})
    self._signature = signature
    self._signature_outputs_as_dict = signature_outputs_as_dict
    self._output_key = output_key
    # TODO(b/142213824): Remove setting shapes when shape inference works.
    if output_shape:
      # Autograph chokes on _convert_nest_to_shapes(), so we call it here
      # and not from within call().
      self._output_shape = data_structures.NoDependency(
          _convert_nest_to_shapes(output_shape))

    self._load_options = load_options
    self._func = load_module(handle, tags, self._load_options)
    self._has_training_argument = func_has_training_argument(self._func)
    self._is_hub_module_v1 = getattr(self._func, "_is_hub_module_v1", False)

    # Update with the defaults when using legacy TF1 Hub format.
    if self._is_hub_module_v1:
      self._signature = self._signature or "default"
      if not self._signature_outputs_as_dict:
        self._output_key = self._output_key or "default"
    # More validity checks.
    if self._signature and (bool(self._output_key is not None)
                            == bool(self._signature_outputs_as_dict)):
      raise ValueError("When using a signature, either output_key or "
                       "signature_outputs_as_dict=True should be set.")
    if not self._signature and self._signature_outputs_as_dict:
      raise ValueError("signature_outputs_as_dict is only valid if specifying "
                       "a signature (or using a legacy TF1 Hub format).")

    self._callable = self._get_callable()
    self._setup_layer(trainable, **kwargs)

  def _setup_layer(self, trainable=False, **kwargs):
    """Constructs keras layer with relevant weights and losses."""
    # Initialize an empty layer, then add_weight() etc. as needed.
    super(KerasLayer, self).__init__(trainable=trainable, **kwargs)

    # Add trainable and non-trainable weights from the callable.
    if hasattr(self._func, "trainable_variables"):
      for v in self._func.trainable_variables:
        self._add_existing_weight(v, trainable=True)
      trainable_variables = {id(v) for v in self._func.trainable_variables}
    else:
      trainable_variables = set()
    if hasattr(self._func, "variables"):
      for v in self._func.variables:
        if id(v) not in trainable_variables:
          self._add_existing_weight(v, trainable=False)

    # Forward the callable's regularization losses (if any).
    if hasattr(self._func, "regularization_losses"):
      for l in self._func.regularization_losses:
        if not callable(l):
          raise ValueError(
              "hub.KerasLayer(obj) expects obj.regularization_losses to be an "
              "iterable of callables, each returning a scalar loss term.")
        self.add_loss(self._call_loss_if_trainable(l))  # Supports callables.

  def _add_existing_weight(self, weight, trainable=None):
    """Calls add_weight() to register but not create an existing weight."""
    if trainable is None: trainable = weight.trainable
    self.add_weight(name=weight.name, shape=weight.shape, dtype=weight.dtype,
                    trainable=trainable, getter=lambda *_, **__: weight)

  def _call_loss_if_trainable(self, loss):
    """Returns `loss` conditioned on whether this layer is trainable."""
    return lambda: loss() if self.trainable else 0.

  def call(self, inputs, training=None):
    # These checks happen here and not in __init__, because self.trainable is
    # a mutable public attribute.
    self._check_trainability()

    # We basically want to call this...
    args = []
    kwargs = self._arguments.copy()
    if self._signature and isinstance(inputs, dict):
      kwargs.update(inputs)
    else:
      args.append(inputs)
    f = functools.partial(self._callable, *args, **kwargs)
    # ...but we may also have to pass a Python boolean for `training`, which
    # is the logical "and" of this layer's trainability and what the surrounding
    # model is doing (analogous to tf.keras.layers.BatchNormalization in TF2).
    # For the latter, we have to look in two places: the `training` argument,
    # or else Keras' global `learning_phase`, which might actually be a tensor.
    if not self._has_training_argument:
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

    # Unwrap dicts returned by signatures.
    if self._output_key:
      if not isinstance(result, dict):
        raise ValueError("Specifying `output_key` is forbidden if output "
                         "type %s is not a dict." % type(result))
      if self._output_key not in result:
        raise ValueError(
            "KerasLayer output does not contain the output key %s "
            "(available: %s)." % (self._output_key, result.keys()))
      result = result[self._output_key]

    # TODO(b/142213824): Remove setting shapes when shape inference works.
    result = self._apply_output_shape_if_set(inputs, result)
    return result

  def _check_trainability(self):
    """Raises or logs errors for unuspported uses of trainable=True."""
    if not self.trainable: return  # Nothing to do.

    # Training is only supported when calling a reusable TF2 SavedModel through
    # its @tf.function __call__. Trying to train through a signature is likely
    # to go wrong beyond the most simple cases due to a number of pitfalls:
    # - No good support for train vs inference mode. TF1 Hub format used
    #   graph versions identified by tags, but this was not a general
    #   standard for SavedModels, and TF2 can no longer save with tags.
    # - No support for update ops. TF1 Hub format had them in the UPDATE_OPS
    #   collection, but collections are no longer loaded in TF2. General
    #   SavedModel signatures had no support for them.
    # - No support for regularization losses (same story).
    # - A SavedModel without @tf.function __call__ will likely also not
    #   provide a trainable_variables attribute.
    if self._is_hub_module_v1:
      raise ValueError(
          "Setting hub.KerasLayer.trainable = True is unsupported when "
          "loading from the TF1 Hub format.")
    elif self._signature:
      raise ValueError(
          "Setting hub.KerasLayer.trainable = True is unsupported when "
          "calling a SavedModel signature.")
    # Having zero trainable variables in an otherwise trainable model
    # is suspicious but may be valid as a boundary case, so we just log,
    # but at most once per layer instance.
    if not self.trainable_weights:
      if not hasattr(self, "_already_logged_trainable_with_zero_weights"):
        logging.error(
            "hub.KerasLayer is trainable but has zero trainable weights.")
        setattr(self, "_already_logged_trainable_with_zero_weights", True)

  def _get_callable(self):
    """Returns a callable object."""
    if callable(self._func) and not self._signature:
      return self._func
    if not hasattr(self._func, "signatures"):
      if self._signature:  # Assuming the user intended to use a signature.
        raise ValueError("Loaded object has no signatures.")
      else:  # Assuming the user intended to use a callable SavedModel.
        raise ValueError(
            "Loaded object is not callable and has no signatures.")
    if self._signature is None:
      raise ValueError("Signature name has to be specified for non-callable "
                       "saved models (if not legacy TF1 Hub format).")
    if self._signature not in self._func.signatures:
      raise ValueError("Unknown signature %s in %s (available signatures: %s)."
                       % (self._signature, self._handle, self._func.signatures))
    f = self._func.signatures[self._signature]
    if not callable(f):
      raise ValueError("Internal error: signature %s is not callable in %s" %
                       (self._signature, self._handle))
    return f

  def _apply_output_shape_if_set(self, inputs, outputs):
    if not hasattr(self, "_output_shape"):
      return outputs
    # Traverse the nest and turn shape-like tuples into tf.TensorShapes,
    # or else map_structure below would try to recurse into them.
    output_shape = getattr(self, "_output_shape")
    batch_size = tf.nest.flatten(inputs)[0].shape[0]
    def _inplace_set_shape(tensor, shape):
      tensor.set_shape(tf.TensorShape(batch_size).concatenate(shape))
    tf.nest.map_structure(_inplace_set_shape, outputs, output_shape)
    return outputs

  def get_config(self):
    """Returns a serializable dict of keras layer configuration parameters."""
    config = super(KerasLayer, self).get_config()
    if not isinstance(self._handle, six.string_types):
      # Need to raise this type in order for tf.saved_model.save() to fall back
      # to not using config, instead of crashing.
      # TODO(b/134528831): Reconsider the usability implications.
      raise NotImplementedError(
          "Can only generate a valid config for `hub.KerasLayer(handle, ...)`"
          "that uses a string `handle`.\n\n"
          "Got `type(handle)`: {}".format(type(self._handle)))
    config["handle"] = self._handle

    if hasattr(self, "_output_shape"):
      output_shape = _convert_nest_from_shapes(self._output_shape)
      try:
        json.dumps(output_shape)
      except TypeError:
        raise ValueError(
            "hub.KerasLayer(..., output_shape=) is not json-serializable.\n"
            "Got value: {}".format(output_shape))
      config["output_shape"] = output_shape

    if self._arguments:
      # Raise clear errors for non-serializable arguments.
      for key, value in self._arguments.items():
        try:
          json.dumps(value)
        except TypeError:
          raise ValueError(
              "`hub.KerasLayer(..., arguments)` contains non json-serializable"
              "values in key: {}".format(key))
      config["arguments"] = self._arguments

    if self._signature:
      config["signature"] = self._signature
    if self._output_key:
      config["output_key"] = self._output_key
    if self._signature_outputs_as_dict:
      config["signature_outputs_as_dict"] = self._signature_outputs_as_dict

    # self._load_options is not stored in the config. Instead, the load
    # options passed at the time when this layer gets reloaded from its config
    # are applied to its own loading as well. That is because the only
    # load option available at this time (July 2020) is
    # `experimental_io_device`, which relates to the loading environment,
    # and not to the interpretation of the loaded SavedModel.

    return config

  @property
  def resolved_object(self):
    """Returns the callable object to which `handle` resolved in `__init__`."""
    return self._func


def _convert_nest_to_shapes(x):
  """In a nest, converts raw tuples/lists of int or None to tf.TensorShape."""
  # A dict is certainly a container and not a shape. We need to handle
  # it first and not try construct a TensorShape from its keys.
  if isinstance(x, dict):
    return type(x)([(k, _convert_nest_to_shapes(v)) for k, v in x.items()])
  # Anything else might be already a TensorShape, a tuple that converts
  # to a TensorShape, or a sequence that needs further recursion.
  try:
    return tf.TensorShape(x)
  except TypeError:
    pass  # Will try parsing as a container instead.
  if isinstance(x, (list, tuple)):
    return type(x)([_convert_nest_to_shapes(v) for v in x])
  else:
    raise TypeError("Cannot convert to nest of TensorShapes, "
                    "found none of TensorShape, dict, list, tuple: %r" % x)


def _convert_nest_from_shapes(x):
  """Converts a nest of tf.TensorShape to raw tuples of int or None."""
  def _shape_as_tuple(x):
    assert isinstance(x, tf.TensorShape)
    return tuple(x.as_list())
  return tf.nest.map_structure(_shape_as_tuple, x)


def load_module(handle, tags=None, load_options=None):
  if callable(handle):
    if tags is not None:
      raise ValueError("Passing a callable handle is mutually exclusive "
                       "with setting tags.")
    if load_options is not None:
      raise ValueError("Passing a callable handle is mutually exclusive "
                       "with setting load_options.")
    return handle
  else:
    try:
      # pylint: disable=g-import-not-at-top
      # pylint: disable=g-direct-tensorflow-import
      from tensorflow.python.saved_model import load_context
      set_load_options = load_options or load_context.get_load_options()
    except ImportError:
      set_load_options = load_options
    return module_v2.load(handle, tags=tags, options=set_load_options)


def func_has_training_argument(func):
  """Checks whether saved model has a `training` argument."""
  if not callable(func):
    return False
  fullargspec = tf_inspect.getfullargspec(func.__call__)
  return ("training" in fullargspec.args or
          "training" in fullargspec.kwonlyargs)
