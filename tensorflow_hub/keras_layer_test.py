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
"""Tests for tensorflow_hub.keras_layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow_hub import test_utils


# NOTE: A Hub-style SavedModel can either be constructed manually, or by
# relying on tf.saved_model.save(keras_model, ...) to put in the expected
# endpoints. The following _save*model() helpers offer a save_from_keras
# argument to select, and tests should strive to exercise both.
# The big exception are SavedModels with hyperparameters: There is no generic
# helper code yet to bridge between optional tensor inputs and properties
# in Keras model objects.


# A series of code changes implemented the necessary Keras functionality
# up to TF 2.0.0-beta1. For this test to work, we need them up to and including
# https://github.com/tensorflow/tensorflow/commit/3dc3b5df5f87ac0c460583eebc7d845e33138d2b
# However, this is not easy to test for, so we're piggybacking here on the
# one official API change from that series, found in the slightly older
# https://github.com/tensorflow/tensorflow/commit/eff4ae822a08355b4a15b638148129d348b985a4#diff-1088d4bbbe00f5b39b923b4527e93790
def _skip_if_keras_save_too_old(test_case):
  if not hasattr(tf.keras.layers.InputSpec, "get_config"):
    test_case.skipTest(
        "Your TensorFlow version (%s) looks too old for creating reusable "
        "SavedModels in Keras model saving." % tf.__version__)


def _skip_if_no_tf_asset(test_case):
  if not hasattr(tf.saved_model, "Asset"):
    test_case.skipTest(
        "Your TensorFlow version (%s) looks too old for creating SavedModels "
        " with assets." % tf.__version__)


def _json_cycle(x):
  return json.loads(json.dumps(x))


def _save_half_plus_one_model(export_dir, save_from_keras=False):
  """Writes Hub-style SavedModel to compute y = wx + 1, with w trainable."""
  inp = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
  times_w = tf.keras.layers.Dense(
      units=1,
      kernel_initializer=tf.keras.initializers.Constant([[0.5]]),
      kernel_regularizer=tf.keras.regularizers.l2(0.01),
      use_bias=False)
  plus_1 = tf.keras.layers.Dense(
      units=1,
      kernel_initializer=tf.keras.initializers.Constant([[1.0]]),
      bias_initializer=tf.keras.initializers.Constant([1.0]),
      trainable=False)
  outp = plus_1(times_w(inp))
  model = tf.keras.Model(inp, outp)

  if save_from_keras:
    tf.saved_model.save(model, export_dir)
    return

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, 1), dtype=tf.float32)])
  def call_fn(inputs):
    return model(inputs, training=False)

  obj = tf.train.Checkpoint()
  obj.__call__ = call_fn
  obj.variables = model.trainable_variables + model.non_trainable_variables
  assert len(obj.variables) == 3, "Expect 2 kernels and 1 bias."
  obj.trainable_variables = [times_w.kernel]
  assert(len(model.losses) == 1), "Expect 1 regularization loss."
  obj.regularization_losses = [
      tf.function(lambda: model.losses[0], input_signature=[])]
  tf.saved_model.save(obj, export_dir)


def _tensors_names_set(tensor_sequence):
  """Converts tensor sequence to a set of tensor references."""
  # Tensor name stands as a proxy for the uniqueness of the tensors.
  # In TensorFlow 2.x one can use the `experimental_ref` method, but it is not
  # available in older TF versions.
  return {t.name for t in tensor_sequence}


def _save_batch_norm_model(export_dir, save_from_keras=False):
  """Writes a Hub-style SavedModel with a batch norm layer."""
  inp = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
  bn = tf.keras.layers.BatchNormalization(momentum=0.8)
  outp = bn(inp)
  model = tf.keras.Model(inp, outp)

  if save_from_keras:
    tf.saved_model.save(model, export_dir)
    return

  @tf.function
  def call_fn(inputs, training=False):
    return model(inputs, training=training)
  for training in (True, False):
    call_fn.get_concrete_function(tf.TensorSpec((None, 1), tf.float32),
                                  training=training)

  obj = tf.train.Checkpoint()
  obj.__call__ = call_fn
  # Test assertions pick up variables by their position here.
  obj.trainable_variables = [bn.beta, bn.gamma]
  assert _tensors_names_set(obj.trainable_variables) == _tensors_names_set(
      model.trainable_variables)
  obj.variables = [bn.beta, bn.gamma, bn.moving_mean, bn.moving_variance]
  assert _tensors_names_set(obj.variables) == _tensors_names_set(
      model.trainable_variables + model.non_trainable_variables)
  obj.regularization_losses = []
  assert not model.losses
  tf.saved_model.save(obj, export_dir)


def _get_batch_norm_vars(imported):
  """Returns the 4 variables of an imported batch norm model in sorted order."""
  expected_suffixes = ["beta", "gamma", "moving_mean", "moving_variance"]
  variables = sorted(imported.variables, key=lambda v: v.name)
  names = [v.name for v in variables]
  assert len(variables) == 4
  assert all(name.endswith(suffix + ":0")
             for name, suffix in zip(names, expected_suffixes))
  return variables


def _save_model_with_hparams(export_dir):
  """Writes a Hub-style SavedModel to compute y = ax + b with hparams a, b."""
  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
      tf.TensorSpec(shape=(), dtype=tf.float32),
      tf.TensorSpec(shape=(), dtype=tf.float32)])
  def call_fn(x, a=1., b=0.):
    return tf.add(tf.multiply(a, x), b)

  obj = tf.train.Checkpoint()
  obj.__call__ = call_fn
  tf.saved_model.save(obj, export_dir)


def _save_model_with_custom_attributes(export_dir, temp_dir,
                                       save_from_keras=False):
  """Writes a Hub-style SavedModel with a custom attributes."""
  # Calling the module parses an integer.
  f = lambda a: tf.strings.to_number(a, tf.int64)
  if save_from_keras:
    inp = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
    outp = tf.keras.layers.Lambda(f)(inp)
    model = tf.keras.Model(inp, outp)
  else:
    model = tf.train.Checkpoint()
    model.__call__ = tf.function(
        input_signature=[tf.TensorSpec(shape=(None, 1), dtype=tf.string)])(f)

  # Running on the `sample_input` file yields the `sample_output` value.
  asset_source_file_name = os.path.join(temp_dir, "number.txt")
  tf.io.gfile.makedirs(temp_dir)
  with tf.io.gfile.GFile(asset_source_file_name, "w") as f:
    f.write("12345\n")
  model.sample_input = tf.saved_model.Asset(asset_source_file_name)
  model.sample_output = tf.Variable([[12345]], dtype=tf.int64)

  # Save model and invalidate the original asset file name.
  tf.saved_model.save(model, export_dir)
  tf.io.gfile.remove(asset_source_file_name)
  return export_dir


def _save_model_with_dict_input_output(export_dir):
  """Writes SavedModel using dicts to compute x+y, x+2y and maybe x-y."""
  @tf.function
  def call_fn(d, return_dict=False):
    x = d["x"]
    y = d["y"]
    sigma = tf.concat([tf.add(x, y), tf.add(x, 2*y)], axis=-1)
    if return_dict:
      return dict(sigma=sigma, delta=tf.subtract(x, y))
    else:
      return sigma
  # Trigger traces.
  d_spec = dict(x=tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                y=tf.TensorSpec(shape=(None, 1), dtype=tf.float32))
  for return_dict in (False, True):
    call_fn.get_concrete_function(d_spec, return_dict=return_dict)

  obj = tf.train.Checkpoint()
  obj.__call__ = call_fn
  tf.saved_model.save(obj, export_dir)


def _save_model_with_obscurely_shaped_list_output(export_dir):
  """Writes SavedModel with hard-to-predict output shapes."""
  def broadcast_obscurely_to(input, shape):
    """Like tf.broadcast_to(), but hostile to static shape propagation."""
    obscured_shape = tf.cast(tf.cast(shape, tf.float32)
                             # Add small random noise that gets rounded away.
                             + 0.1*tf.sin(tf.random.uniform((), -3, +3)) + 0.3,
                             tf.int32)
    return tf.broadcast_to(input, obscured_shape)

  @tf.function(
      input_signature=[tf.TensorSpec(shape=(None, 1), dtype=tf.float32)])
  def call_fn(x):
    # For each batch element x, the three outputs are
    #   value x with shape (1)
    #   value 2*x broadcast to shape (2,2)
    #   value 3*x broadcast to shape (3,3,3)
    batch_size = tf.shape(x)[0]
    return [broadcast_obscurely_to(tf.reshape(i*x, [batch_size] + [1]*i),
                                   tf.concat([[batch_size], [i]*i], axis=0))
            for i in range(1, 4)]

  obj = tf.train.Checkpoint()
  obj.__call__ = call_fn
  tf.saved_model.save(obj, export_dir)


class KerasTest(tf.test.TestCase, parameterized.TestCase):
  """Tests KerasLayer in an all-Keras environment."""

  @parameterized.named_parameters(("SavedRaw", False), ("SavedFromKeras", True))
  def testHalfPlusOneRetraining(self, save_from_keras):
    if save_from_keras: _skip_if_keras_save_too_old(self)
    # Import the half-plus-one model into a consumer model.
    export_dir = os.path.join(self.get_temp_dir(), "half-plus-one")
    _save_half_plus_one_model(export_dir, save_from_keras=save_from_keras)
    inp = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
    imported = hub.KerasLayer(export_dir, trainable=True)
    outp = imported(inp)
    model = tf.keras.Model(inp, outp)
    # The consumer model computes y = x/2 + 1 as expected.
    self.assertAllEqual(
        model(np.array([[0.], [8.], [10.], [12.]], dtype=np.float32)),
        np.array([[1.], [5.], [6.], [7.]], dtype=np.float32))
    self.assertAllEqual(model.losses, np.array([0.0025], dtype=np.float32))
    # The kernel weight is trainable but the bias is not.
    self.assertEqual(len(model.trainable_weights), 1)
    self.assertEqual(model.trainable_weights[0].shape.rank, 2)  # Kernel w.
    self.assertEqual(len(model.non_trainable_weights), 2)
    self.assertCountEqual([v.shape.rank for v in model.non_trainable_weights],
                          [2, 1])  # Kernel and bias from the plus_1 layer.
    self.assertNoCommonElements(_tensors_names_set(model.trainable_weights),
                                _tensors_names_set(model.non_trainable_weights))
    # Retrain on y = x/2 + 6 for x near 10.
    # (Console output should show loss below 0.2.)
    model.compile(tf.keras.optimizers.SGD(0.002),
                  "mean_squared_error", run_eagerly=True)
    x = [[9.], [10.], [11.]] * 10
    y = [[xi[0]/2. + 6] for xi in x]
    model.fit(np.array(x), np.array(y), batch_size=len(x), epochs=10, verbose=2)
    # The bias is non-trainable and has to stay at 1.0.
    self.assertAllEqual(model(np.array([[0.]], dtype=np.float32)),
                        np.array([[1.]], dtype=np.float32))
    # To compensate, the kernel weight will grow to almost 1.0.
    self.assertAllClose(model(np.array([[10.]], dtype=np.float32)),
                        np.array([[11.]], dtype=np.float32),
                        atol=0.0, rtol=0.03)
    self.assertAllClose(model.losses, np.array([0.01], dtype=np.float32),
                        atol=0.0, rtol=0.06)

  @parameterized.named_parameters(("SavedRaw", False), ("SavedFromKeras", True))
  def testRegularizationLoss(self, save_from_keras):
    if save_from_keras: _skip_if_keras_save_too_old(self)
    # Import the half-plus-one model into a consumer model.
    export_dir = os.path.join(self.get_temp_dir(), "half-plus-one")
    _save_half_plus_one_model(export_dir, save_from_keras=save_from_keras)
    inp = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
    imported = hub.KerasLayer(export_dir, trainable=False)
    outp = imported(inp)
    model = tf.keras.Model(inp, outp)
    # When untrainable, the layer does not contribute regularization losses.
    self.assertAllEqual(model.losses, np.array([0.], dtype=np.float32))
    # When trainable (even set after the fact), the layer forwards its losses.
    imported.trainable = True
    self.assertAllEqual(model.losses, np.array([0.0025], dtype=np.float32))
    # This can be toggled repeatedly.
    imported.trainable = False
    self.assertAllEqual(model.losses, np.array([0.], dtype=np.float32))
    imported.trainable = True
    self.assertAllEqual(model.losses, np.array([0.0025], dtype=np.float32))

  @parameterized.named_parameters(("SavedRaw", False), ("SavedFromKeras", True))
  def testBatchNormRetraining(self, save_from_keras):
    """Tests imported batch norm with trainable=True."""
    if save_from_keras: _skip_if_keras_save_too_old(self)
    export_dir = os.path.join(self.get_temp_dir(), "batch-norm")
    _save_batch_norm_model(export_dir, save_from_keras=save_from_keras)
    inp = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
    imported = hub.KerasLayer(export_dir, trainable=True)
    var_beta, var_gamma, var_mean, var_variance = _get_batch_norm_vars(imported)
    outp = imported(inp)
    model = tf.keras.Model(inp, outp)
    # Retrain the imported batch norm layer on a fixed batch of inputs,
    # which has mean 12.0 and some variance of a less obvious value.
    # The module learns scale and offset parameters that achieve the
    # mapping x --> 2*x for the observed mean and variance.
    model.compile(tf.keras.optimizers.SGD(0.1),
                  "mean_squared_error", run_eagerly=True)
    x = [[11.], [12.], [13.]]
    y = [[2*xi[0]] for xi in x]
    model.fit(np.array(x), np.array(y), batch_size=len(x), epochs=100)
    self.assertAllClose(var_mean.numpy(), np.array([12.0]))
    self.assertAllClose(var_beta.numpy(), np.array([24.0]))
    self.assertAllClose(model(np.array(x, np.float32)), np.array(y))
    # Evaluating the model operates batch norm in inference mode:
    # - Batch statistics are ignored in favor of aggregated statistics,
    #   computing x --> 2*x independent of input distribution.
    # - Update ops are not run, so this doesn't change over time.
    for _ in range(100):
      self.assertAllClose(model(np.array([[10.], [20.], [30.]], np.float32)),
                          np.array([[20.], [40.], [60.]]))
    self.assertAllClose(var_mean.numpy(), np.array([12.0]))
    self.assertAllClose(var_beta.numpy(), np.array([24.0]))

  @parameterized.named_parameters(("SavedRaw", False), ("SavedFromKeras", True))
  def testBatchNormFreezing(self, save_from_keras):
    """Tests imported batch norm with trainable=False."""
    if save_from_keras: _skip_if_keras_save_too_old(self)
    export_dir = os.path.join(self.get_temp_dir(), "batch-norm")
    _save_batch_norm_model(export_dir, save_from_keras=save_from_keras)
    inp = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
    imported = hub.KerasLayer(export_dir, trainable=False)
    var_beta, var_gamma, var_mean, var_variance = _get_batch_norm_vars(imported)
    dense = tf.keras.layers.Dense(
        units=1,
        kernel_initializer=tf.keras.initializers.Constant([[1.5]]),
        use_bias=False)
    outp = dense(imported(inp))
    model = tf.keras.Model(inp, outp)
    # Training the model to x --> 2*x leaves the batch norm layer entirely
    # unchanged (both trained beta&gamma and aggregated mean&variance).
    self.assertAllClose(var_beta.numpy(), np.array([0.0]))
    self.assertAllClose(var_gamma.numpy(), np.array([1.0]))
    self.assertAllClose(var_mean.numpy(), np.array([0.0]))
    self.assertAllClose(var_variance.numpy(), np.array([1.0]))
    model.compile(tf.keras.optimizers.SGD(0.1),
                  "mean_squared_error", run_eagerly=True)
    x = [[1.], [2.], [3.]]
    y = [[2*xi[0]] for xi in x]
    model.fit(np.array(x), np.array(y), batch_size=len(x), epochs=20)
    self.assertAllClose(var_beta.numpy(), np.array([0.0]))
    self.assertAllClose(var_gamma.numpy(), np.array([1.0]))
    self.assertAllClose(var_mean.numpy(), np.array([0.0]))
    self.assertAllClose(var_variance.numpy(), np.array([1.0]))
    self.assertAllClose(model(np.array(x, np.float32)), np.array(y))

  @parameterized.named_parameters(("SavedRaw", False), ("SavedFromKeras", True))
  def testCustomAttributes(self, save_from_keras):
    """Tests custom attributes (Asset and Variable) on a SavedModel."""
    if save_from_keras: _skip_if_keras_save_too_old(self)
    _skip_if_no_tf_asset(self)
    base_dir = os.path.join(self.get_temp_dir(), "custom-attributes")
    export_dir = os.path.join(base_dir, "model")
    temp_dir = os.path.join(base_dir, "scratch")
    _save_model_with_custom_attributes(export_dir, temp_dir,
                                       save_from_keras=save_from_keras)
    imported = hub.KerasLayer(export_dir)
    expected_outputs = imported.resolved_object.sample_output.value().numpy()
    asset_path = imported.resolved_object.sample_input.asset_path.numpy()
    with tf.io.gfile.GFile(asset_path) as f:
      inputs = tf.constant([[f.read()]], dtype=tf.string)
    actual_outputs = imported(inputs).numpy()
    self.assertAllEqual(expected_outputs, actual_outputs)

  @parameterized.named_parameters(("NoOutputShapes", False),
                                  ("WithOutputShapes", True))
  def testInputOutputDict(self, pass_output_shapes):
    """Tests use of input/output dicts."""
    # Create a SavedModel to compute sigma=[x+y, x+2y] and maybe delta=x-y.
    export_dir = os.path.join(self.get_temp_dir(), "with-dicts")
    _save_model_with_dict_input_output(export_dir)
    # Build a Model from it using Keras' "functional" API.
    x_in = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
    y_in = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
    dict_in = dict(x=x_in, y=y_in)
    kwargs = dict(arguments=dict(return_dict=True))  # For the SavedModel.
    if pass_output_shapes:
      # Shape inference works without this, but we pass it anyways to exercise
      # that code path and see that map_structure is called correctly
      # and calls Tensor.set_shape() with compatible values.
      kwargs["output_shape"] = dict(sigma=(2,), delta=(1,))
    imported = hub.KerasLayer(export_dir, **kwargs)
    dict_out = imported(dict_in)
    delta_out = dict_out["delta"]
    sigma_out = dict_out["sigma"]
    concat_out = tf.keras.layers.concatenate([delta_out, sigma_out])
    model = tf.keras.Model(dict_in, [delta_out, sigma_out, concat_out])
    # Test the model.
    x = np.array([[11.], [22.], [33.]], dtype=np.float32)
    y = np.array([[1.], [2.], [3.]], dtype=np.float32)
    outputs = model(dict(x=x, y=y))
    self.assertLen(outputs, 3)
    delta, sigma, concat = [x.numpy() for x in outputs]
    self.assertAllClose(delta,
                        np.array([[10.], [20.], [30.]]))
    self.assertAllClose(sigma,
                        np.array([[12., 13.], [24., 26.], [36., 39.]]))
    self.assertAllClose(
        concat,
        np.array([[10., 12., 13.], [20., 24., 26.], [30., 36., 39.]]))
    # Test round-trip through config.
    config = imported.get_config()
    new_layer = hub.KerasLayer.from_config(_json_cycle(config))
    if pass_output_shapes:
      self.assertEqual(new_layer._output_shape, imported._output_shape)
    else:
      self.assertFalse(hasattr(new_layer, "_output_shape"))

  @parameterized.named_parameters(("NoOutputShapes", False),
                                  ("WithOutputShapes", True))
  def testOutputShapeList(self, pass_output_shapes):
    export_dir = os.path.join(self.get_temp_dir(), "obscurely-shaped")
    _save_model_with_obscurely_shaped_list_output(export_dir)

    kwargs = {}
    if pass_output_shapes:
      kwargs["output_shape"] = [[1], [2, 2], [3, 3, 3]]
    inp = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
    imported = hub.KerasLayer(export_dir, **kwargs)
    outp = imported(inp)
    model = tf.keras.Model(inp, outp)

    x = np.array([[1.], [10.]], dtype=np.float32)
    outputs = model(x)
    self.assertLen(outputs, 3)
    single, double, triple = [x.numpy() for x in outputs]
    # The outputs above are eager Tensors with concrete values,
    # so they always have a fully defined shape. However, running
    # without crash verifies that no incompatible shapes were set.
    # See EstimatorTest below for graph-mode Tensors.
    self.assertAllClose(single, np.array([[1.], [10.]]))
    self.assertAllClose(double, np.array([[[2., 2.], [2., 2.]],
                                          [[20., 20.], [20., 20.]]]))
    self.assertAllClose(triple, np.array(
        [[[[3., 3., 3.], [3., 3., 3.], [3., 3., 3.]],
          [[3., 3., 3.], [3., 3., 3.], [3., 3., 3.]],
          [[3., 3., 3.], [3., 3., 3.], [3., 3., 3.]]],
         [[[30., 30., 30.], [30., 30., 30.], [30., 30., 30.]],
          [[30., 30., 30.], [30., 30., 30.], [30., 30., 30.]],
          [[30., 30., 30.], [30., 30., 30.], [30., 30., 30.]]]]))
    # Test round-trip through config.
    config = imported.get_config()
    new_layer = hub.KerasLayer.from_config(_json_cycle(config))
    if pass_output_shapes:
      self.assertEqual(new_layer._output_shape, imported._output_shape)
    else:
      self.assertFalse(hasattr(new_layer, "_output_shape"))

  @parameterized.named_parameters(("SavedRaw", False), ("SavedFromKeras", True))
  def testComputeOutputShape(self, save_from_keras):
    if save_from_keras: _skip_if_keras_save_too_old(self)
    export_dir = os.path.join(self.get_temp_dir(), "half-plus-one")
    _save_half_plus_one_model(export_dir, save_from_keras=save_from_keras)
    layer = hub.KerasLayer(export_dir, output_shape=[1])
    self.assertEqual([10, 1],
                     layer.compute_output_shape(tuple([10, 1])).as_list())
    layer.get_config()

  @parameterized.named_parameters(("SavedRaw", False), ("SavedFromKeras", True))
  def testGetConfigFromConfig(self, save_from_keras):
    if save_from_keras: _skip_if_keras_save_too_old(self)
    export_dir = os.path.join(self.get_temp_dir(), "half-plus-one")
    _save_half_plus_one_model(export_dir, save_from_keras=save_from_keras)
    layer = hub.KerasLayer(export_dir)
    in_value = np.array([[10.0]], dtype=np.float32)
    result = layer(in_value).numpy()

    config = layer.get_config()
    new_layer = hub.KerasLayer.from_config(_json_cycle(config))
    new_result = new_layer(in_value).numpy()
    self.assertEqual(result, new_result)

  def testGetConfigFromConfigWithHParams(self):
    if tf.__version__ == "2.0.0-alpha0":
      self.skipTest("b/127938157 broke use of default hparams")
    export_dir = os.path.join(self.get_temp_dir(), "with-hparams")
    _save_model_with_hparams(export_dir)  # Has no `save_from_keras` arg.
    layer = hub.KerasLayer(export_dir, arguments=dict(a=10.))  # Leave b=0.
    in_value = np.array([[1.], [2.], [3.]], dtype=np.float32)
    expected_result = np.array([[10.], [20.], [30.]], dtype=np.float32)
    result = layer(in_value).numpy()
    self.assertAllEqual(expected_result, result)

    config = layer.get_config()
    new_layer = hub.KerasLayer.from_config(_json_cycle(config))
    new_result = new_layer(in_value).numpy()
    self.assertAllEqual(result, new_result)

  @parameterized.named_parameters(("SavedRaw", False), ("SavedFromKeras", True))
  def testSaveModelConfig(self, save_from_keras):
    if save_from_keras: _skip_if_keras_save_too_old(self)
    export_dir = os.path.join(self.get_temp_dir(), "half-plus-one")
    _save_half_plus_one_model(export_dir, save_from_keras=save_from_keras)

    model = tf.keras.Sequential([hub.KerasLayer(export_dir)])
    in_value = np.array([[10.]], dtype=np.float32)
    result = model(in_value).numpy()

    json_string = model.to_json()
    new_model = tf.keras.models.model_from_json(
        json_string, custom_objects={"KerasLayer": hub.KerasLayer})
    new_result = new_model(in_value).numpy()
    self.assertEqual(result, new_result)


class EstimatorTest(tf.test.TestCase, parameterized.TestCase):
  """Tests use of KerasLayer in an Estimator's model_fn."""

  def _half_plus_one_model_fn(self, features, labels, mode, params):
    inp = features  # This estimator takes a single feature, not a dict.
    imported = hub.KerasLayer(params["hub_module"],
                              trainable=params["hub_trainable"])
    model = tf.keras.Sequential([imported])
    outp = model(inp, training=(mode == tf.estimator.ModeKeys.TRAIN))
    # https://www.tensorflow.org/alpha/guide/migration_guide#using_a_custom_model_fn
    # recommends model.get_losses_for() instead of model.losses.
    model_losses = model.get_losses_for(None) + model.get_losses_for(inp)
    regularization_loss = tf.add_n(model_losses or [0.0])
    predictions = dict(output=outp, regularization_loss=regularization_loss)

    total_loss = None
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
      total_loss = tf.add(
          tf.compat.v1.losses.mean_squared_error(labels, outp),
          regularization_loss)

    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.002)
      train_op = optimizer.minimize(
          total_loss, var_list=model.trainable_variables,
          global_step=tf.compat.v1.train.get_or_create_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode, predictions=predictions, loss=total_loss, train_op=train_op)

  @parameterized.named_parameters(("SavedRaw", False), ("SavedFromKeras", True))
  def testHalfPlusOneRetraining(self, save_from_keras):
    if save_from_keras: _skip_if_keras_save_too_old(self)
    export_dir = os.path.join(self.get_temp_dir(), "half-plus-one")
    _save_half_plus_one_model(export_dir, save_from_keras=save_from_keras)
    estimator = tf.estimator.Estimator(
        model_fn=self._half_plus_one_model_fn,
        params=dict(hub_module=export_dir, hub_trainable=True))

    # The consumer model computes y = x/2 + 1 as expected.
    predictions = next(estimator.predict(
        tf.compat.v1.estimator.inputs.numpy_input_fn(
            np.array([[0.], [8.], [10.], [12.]], dtype=np.float32),
            shuffle=False),
        yield_single_examples=False))
    self.assertAllEqual(predictions["output"],
                        np.array([[1.], [5.], [6.], [7.]], dtype=np.float32))
    self.assertAllEqual(predictions["regularization_loss"],
                        np.array(0.0025, dtype=np.float32))

    # Retrain on y = x/2 + 6 for x near 10.
    # (Console output should show loss below 0.2.)
    x = [[9.], [10.], [11.]] * 10
    y = [[xi[0]/2. + 6] for xi in x]
    estimator.train(
        tf.compat.v1.estimator.inputs.numpy_input_fn(
            np.array(x, dtype=np.float32),
            np.array(y, dtype=np.float32),
            batch_size=len(x), num_epochs=None, shuffle=False),
        steps=10)
    # The bias is non-trainable and has to stay at 1.0.
    # To compensate, the kernel weight will grow to almost 1.0.
    predictions = next(estimator.predict(
        tf.compat.v1.estimator.inputs.numpy_input_fn(
            np.array([[0.], [10.]], dtype=np.float32), shuffle=False),
        yield_single_examples=False))
    self.assertAllEqual(predictions["output"][0],
                        np.array([1.], dtype=np.float32))
    self.assertAllClose(predictions["output"][1],
                        np.array([11.], dtype=np.float32),
                        atol=0.0, rtol=0.03)
    self.assertAllClose(predictions["regularization_loss"],
                        np.array(0.01, dtype=np.float32),
                        atol=0.0, rtol=0.06)

  @parameterized.named_parameters(("SavedRaw", False), ("SavedFromKeras", True))
  def testHalfPlusOneFrozen(self, save_from_keras):
    if save_from_keras: _skip_if_keras_save_too_old(self)
    export_dir = os.path.join(self.get_temp_dir(), "half-plus-one")
    _save_half_plus_one_model(export_dir, save_from_keras=save_from_keras)
    estimator = tf.estimator.Estimator(
        model_fn=self._half_plus_one_model_fn,
        params=dict(hub_module=export_dir, hub_trainable=False))

    # The consumer model computes y = x/2 + 1 as expected.
    predictions = next(estimator.predict(
        tf.compat.v1.estimator.inputs.numpy_input_fn(
            np.array([[0.], [8.], [10.], [12.]], dtype=np.float32),
            shuffle=False),
        yield_single_examples=False))
    self.assertAllEqual(predictions["output"],
                        np.array([[1.], [5.], [6.], [7.]], dtype=np.float32))
    self.assertAllEqual(predictions["regularization_loss"],
                        np.array(0.0, dtype=np.float32))

  def _batch_norm_model_fn(self, features, labels, mode, params):
    inp = features  # This estimator takes a single feature, not a dict.
    imported = hub.KerasLayer(params["hub_module"])
    var_beta, var_gamma, var_mean, var_variance = _get_batch_norm_vars(imported)
    if params["train_batch_norm"]:
      imported.trainable = True
      model = tf.keras.Sequential([imported])
    else:
      imported.trainable = False
      # When not training the batch norm layer, we train this instead:
      dense = tf.keras.layers.Dense(
          units=1,
          kernel_initializer=tf.keras.initializers.Constant([[1.5]]),
          use_bias=False)
      model = tf.keras.Sequential([imported, dense])
    outp = model(inp, training=(mode == tf.estimator.ModeKeys.TRAIN))
    predictions = dict(output=outp,
                       beta=var_beta.value(), gamma=var_gamma.value(),
                       mean=var_mean.value(), variance=var_variance.value())

    # https://www.tensorflow.org/alpha/guide/migration_guide#using_a_custom_model_fn
    # recommends model.get_updates_for() instead of model.updates.
    update_ops = model.get_updates_for(None) + model.get_updates_for(inp)

    loss = None
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
      loss = tf.compat.v1.losses.mean_squared_error(labels, outp)

    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(
            loss, var_list=model.trainable_variables,
            global_step=tf.compat.v1.train.get_or_create_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)

  @parameterized.named_parameters(("SavedRaw", False), ("SavedFromKeras", True))
  def testBatchNormRetraining(self, save_from_keras):
    """Tests imported batch norm with trainable=True."""
    if save_from_keras: _skip_if_keras_save_too_old(self)
    export_dir = os.path.join(self.get_temp_dir(), "batch-norm")
    _save_batch_norm_model(export_dir, save_from_keras=save_from_keras)
    estimator = tf.estimator.Estimator(
        model_fn=self._batch_norm_model_fn,
        params=dict(hub_module=export_dir, train_batch_norm=True))

    # Retrain the imported batch norm layer on a fixed batch of inputs,
    # which has mean 12.0 and some variance of a less obvious value.
    # The module learns scale and offset parameters that achieve the
    # mapping x --> 2*x for the observed mean and variance.
    x = [[11.], [12.], [13.]]
    y = [[2*xi[0]] for xi in x]
    train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        np.array(x, dtype=np.float32),
        np.array(y, dtype=np.float32),
        batch_size=len(x), num_epochs=None, shuffle=False)
    estimator.train(train_input_fn, steps=100)
    predictions = next(estimator.predict(train_input_fn,
                                         yield_single_examples=False))
    self.assertAllClose(predictions["mean"], np.array([12.0]))
    self.assertAllClose(predictions["beta"], np.array([24.0]))
    self.assertAllClose(predictions["output"], np.array(y))

    # Evaluating the model operates batch norm in inference mode:
    # - Batch statistics are ignored in favor of aggregated statistics,
    #   computing x --> 2*x independent of input distribution.
    # - Update ops are not run, so this doesn't change over time.
    predict_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        np.array([[10.], [20.], [30.]], dtype=np.float32),
        batch_size=3, num_epochs=100, shuffle=False)
    for predictions in estimator.predict(predict_input_fn,
                                         yield_single_examples=False):
      self.assertAllClose(predictions["output"],
                          np.array([[20.], [40.], [60.]]))
    self.assertAllClose(predictions["mean"], np.array([12.0]))
    self.assertAllClose(predictions["beta"], np.array([24.0]))

  @parameterized.named_parameters(("SavedRaw", False), ("SavedFromKeras", True))
  def testBatchNormFreezing(self, save_from_keras):
    """Tests imported batch norm with trainable=False."""
    if save_from_keras: _skip_if_keras_save_too_old(self)
    export_dir = os.path.join(self.get_temp_dir(), "batch-norm")
    _save_batch_norm_model(export_dir, save_from_keras=save_from_keras)
    estimator = tf.estimator.Estimator(
        model_fn=self._batch_norm_model_fn,
        params=dict(hub_module=export_dir, train_batch_norm=False))
    x = [[1.], [2.], [3.]]
    y = [[2*xi[0]] for xi in x]
    input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        np.array(x, dtype=np.float32),
        np.array(y, dtype=np.float32),
        batch_size=len(x), num_epochs=None, shuffle=False)
    predictions = next(estimator.predict(input_fn, yield_single_examples=False))
    self.assertAllClose(predictions["beta"], np.array([0.0]))
    self.assertAllClose(predictions["gamma"], np.array([1.0]))
    self.assertAllClose(predictions["mean"], np.array([0.0]))
    self.assertAllClose(predictions["variance"], np.array([1.0]))

    # Training the model to x --> 2*x leaves the batch norm layer entirely
    # unchanged (both trained beta&gamma and aggregated mean&variance).
    estimator.train(input_fn, steps=20)
    predictions = next(estimator.predict(input_fn, yield_single_examples=False))
    self.assertAllClose(predictions["beta"], np.array([0.0]))
    self.assertAllClose(predictions["gamma"], np.array([1.0]))
    self.assertAllClose(predictions["mean"], np.array([0.0]))
    self.assertAllClose(predictions["variance"], np.array([1.0]))
    self.assertAllClose(predictions["output"], np.array(y))

  def _output_shape_list_model_fn(self, features, labels, mode, params):
    inp = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
    kwargs = {}
    if "output_shape" in params:
      kwargs["output_shape"] = params["output_shape"]
    imported = hub.KerasLayer(params["hub_module"], **kwargs)
    outp = imported(inp)
    model = tf.keras.Model(inp, outp)

    out_list = model(features, training=(mode == tf.estimator.ModeKeys.TRAIN))
    for j, out in enumerate(out_list):
      i = j+1  # Sample shapes count from one.
      actual_shape = out.shape.as_list()[1:]  # Without batch size.
      expected_shape = [i]*i if "output_shape" in params else [None]*i
      self.assertEqual(actual_shape, expected_shape)
    predictions = {["one", "two", "three"][i]: out_list[i] for i in range(3)}
    imported.get_config()

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                                      loss=None, train_op=None)

  @parameterized.named_parameters(("NoOutputShapes", False),
                                  ("WithOutputShapes", True))
  def testOutputShapeList(self, pass_output_shapes):
    export_dir = os.path.join(self.get_temp_dir(), "obscurely-shaped")
    _save_model_with_obscurely_shaped_list_output(export_dir)

    params = dict(hub_module=export_dir)
    if pass_output_shapes:
      params["output_shape"] = [[1], [2, 2], [3, 3, 3]]
    estimator = tf.estimator.Estimator(
        model_fn=self._output_shape_list_model_fn,
        params=params)
    x = [[1.], [10.]]
    input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        np.array(x, dtype=np.float32),
        batch_size=len(x), num_epochs=None, shuffle=False)
    predictions = next(estimator.predict(input_fn, yield_single_examples=False))
    single = predictions["one"]
    double = predictions["two"]
    triple = predictions["three"]
    self.assertAllClose(single, np.array([[1.], [10.]]))
    self.assertAllClose(double, np.array([[[2., 2.], [2., 2.]],
                                          [[20., 20.], [20., 20.]]]))
    self.assertAllClose(triple, np.array(
        [[[[3., 3., 3.], [3., 3., 3.], [3., 3., 3.]],
          [[3., 3., 3.], [3., 3., 3.], [3., 3., 3.]],
          [[3., 3., 3.], [3., 3., 3.], [3., 3., 3.]]],
         [[[30., 30., 30.], [30., 30., 30.], [30., 30., 30.]],
          [[30., 30., 30.], [30., 30., 30.], [30., 30., 30.]],
          [[30., 30., 30.], [30., 30., 30.], [30., 30., 30.]]]]))


class KerasLayerTest(tf.test.TestCase, parameterized.TestCase):
  """Unit tests for KerasLayer."""

  @parameterized.named_parameters(
      ("v1_implicit_tags", "hub_module_v1_mini"),
      ("v2_implicit_tags", "saved_model_v2_mini"),
      )
  def test_load_with_defaults(self, module_name):
    inputs, expected_outputs = 10., 11.  # Test modules perform increment op.
    path = test_utils.get_test_data_path(module_name)
    layer = hub.KerasLayer(path)
    output = layer(inputs)
    self.assertEqual(output, expected_outputs)

  @parameterized.parameters(
      ("hub_module_v1_mini", None, None, True),
      ("hub_module_v1_mini", None, None, False),
      ("hub_module_v1_mini", "default", None, True),
      ("hub_module_v1_mini", None, "default", False),
      ("hub_module_v1_mini", "default", "default", False),
      )
  def test_load_legacy_hub_module_v1_with_signature(
      self, module_name, signature, output_key, as_dict):
    inputs, expected_outputs = 10., 11.  # Test modules perform increment op.
    path = test_utils.get_test_data_path(module_name)
    layer = hub.KerasLayer(path, signature=signature, output_key=output_key,
                           signature_outputs_as_dict=as_dict)
    output = layer(inputs)
    if as_dict:
      self.assertEqual(output, {"default": expected_outputs})
    else:
      self.assertEqual(output, expected_outputs)

  @parameterized.parameters(
      ("saved_model_v2_mini", None, None, False),
      ("saved_model_v2_mini", "serving_default", None, True),
      ("saved_model_v2_mini", "serving_default", "output_0", False),
      )
  def test_load_callable_saved_model_v2_with_signature(
      self, module_name, signature, output_key, as_dict):
    inputs, expected_outputs = 10., 11.  # Test modules perform increment op.
    path = test_utils.get_test_data_path(module_name)
    layer = hub.KerasLayer(path, signature=signature, output_key=output_key,
                           signature_outputs_as_dict=as_dict)
    output = layer(inputs)
    if as_dict:
      self.assertIsInstance(output, dict)
      self.assertEqual(output["output_0"], expected_outputs)
    else:
      self.assertEqual(output, expected_outputs)

  @parameterized.parameters(
      ("hub_module_v1_mini", None, None, True),
      ("hub_module_v1_mini", None, None, False),
      ("hub_module_v1_mini", "default", None, True),
      ("hub_module_v1_mini", None, "default", False),
      ("hub_module_v1_mini", "default", "default", False),
      ("saved_model_v2_mini", None, None, False),
      ("saved_model_v2_mini", "serving_default", None, True),
      ("saved_model_v2_mini", "serving_default", "output_0", False),
      )
  def test_keras_layer_get_config(
      self, module_name, signature, output_key, as_dict):
    inputs = 10.  # Test modules perform increment op.
    path = test_utils.get_test_data_path(module_name)
    layer = hub.KerasLayer(path, signature=signature, output_key=output_key,
                           signature_outputs_as_dict=as_dict)
    outputs = layer(inputs)
    config = layer.get_config()
    new_layer = hub.KerasLayer.from_config(_json_cycle(config))
    new_outputs = new_layer(inputs)
    self.assertEqual(outputs, new_outputs)

  def test_keras_layer_fails_if_signature_output_not_specified(self):
    path = test_utils.get_test_data_path("saved_model_v2_mini")
    with self.assertRaisesRegex(
        ValueError, "When using a signature, either output_key or "
        "signature_outputs_as_dict=True should be set."):
      hub.KerasLayer(path, signature="serving_default")

  def test_keras_layer_fails_if_with_outputs_as_dict_but_no_signature(self):
    path = test_utils.get_test_data_path("saved_model_v2_mini")
    with self.assertRaisesRegex(
        ValueError,
        "signature_outputs_as_dict is only valid if specifying a signature *"):
      hub.KerasLayer(path, signature_outputs_as_dict=True)

  def test_keras_layer_fails_if_saved_model_v2_with_tags(self):
    path = test_utils.get_test_data_path("saved_model_v2_mini")
    with self.assertRaises(ValueError):
      hub.KerasLayer(path, signature=None, tags=["train"])

  def test_keras_layer_fails_if_setting_both_output_key_and_as_dict(self):
    path = test_utils.get_test_data_path("hub_module_v1_mini")
    with self.assertRaisesRegex(
        ValueError, "When using a signature, either output_key or "
        "signature_outputs_as_dict=True should be set."):
      hub.KerasLayer(path, signature="default",
                     signature_outputs_as_dict=True, output_key="output")

  def test_keras_layer_fails_if_output_is_not_dict(self):
    path = test_utils.get_test_data_path("saved_model_v2_mini")
    layer = hub.KerasLayer(path, output_key="output_0")
    with self.assertRaisesRegex(
        ValueError, "Specifying `output_key` is forbidden if output type *"):
      layer(10.)

  def test_keras_layer_fails_if_output_key_not_in_layer_outputs(self):
    path = test_utils.get_test_data_path("hub_module_v1_mini")
    layer = hub.KerasLayer(path, output_key="unknown")
    with self.assertRaisesRegex(
        ValueError, "KerasLayer output does not contain the output key*"):
      layer(10.)

  def test_keras_layer_fails_if_hub_module_trainable(self):
    path = test_utils.get_test_data_path("hub_module_v1_mini")
    layer = hub.KerasLayer(path, trainable=True)
    with self.assertRaisesRegex(ValueError, "trainable.*=.*True.*unsupported"):
      layer(10.)

  def test_keras_layer_fails_if_signature_trainable(self):
    path = test_utils.get_test_data_path("saved_model_v2_mini")
    layer = hub.KerasLayer(path, signature="serving_default",
                           signature_outputs_as_dict=True, trainable=True)
    layer.trainable = True
    with self.assertRaisesRegex(ValueError, "trainable.*=.*True.*unsupported"):
      layer(10.)

  def test_keras_layer_logs_if_training_zero_variables(self):
    path = os.path.join(self.get_temp_dir(), "zero-variables")
    _save_model_with_hparams(path)
    layer = hub.KerasLayer(path, trainable=True)
    if hasattr(self, "assertLogs"):  # New in Python 3.4.
      with self.assertLogs(level="ERROR") as logs:
        layer([[10.]])
        layer([[10.]])
      self.assertLen(logs.records, 1)  # Duplicate logging is avoided.
      self.assertRegexpMatches(logs.records[0].msg, "zero trainable weights")
    else:
      # Just test that it runs at all.
      layer([[10.]])
      layer([[10.]])


if __name__ == "__main__":
  # The file under test is not imported if TensorFlow is too old.
  # In such an environment, this test should be silently skipped.
  if hasattr(hub, "KerasLayer"):
    # At this point, we are either in in a late TF1 version or in TF2.
    # In TF1, we need to enable V2-like behavior, notably eager execution.
    # `tf.enable_v2_behavior` seems available and should be preferred.
    # The alternative `tf.enable_eager_behavior` has been around for longer, and
    # will be enabled if `tf.enable_v2_behavior` is not available.
    # In TF2, those enable_*() methods are unnecessary and no longer available.
    if hasattr(tf, "enable_v2_behavior"):
      tf.enable_v2_behavior()
    elif hasattr(tf, "enable_eager_behavior"):
      tf.enable_eager_behavior()
    tf.test.main()
