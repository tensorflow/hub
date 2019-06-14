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

import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def _save_half_plus_one_model(export_dir):
  """Writes a Hub-style SavedModel to compute y = x/2 + 1."""
  inp = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
  dense = tf.keras.layers.Dense(
      units=1,
      kernel_initializer=tf.keras.initializers.Constant([[0.5]]),
      kernel_regularizer=tf.keras.regularizers.l2(0.01),
      bias_initializer=tf.keras.initializers.Constant([1.0]))
  outp = dense(inp)
  model = tf.keras.Model(inp, outp)

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, 1), dtype=tf.float32)])
  def call_fn(inputs):
    return model(inputs, training=False)

  obj = tf.train.Checkpoint()
  obj.__call__ = call_fn
  obj.variables = model.trainable_variables + model.non_trainable_variables
  assert len(obj.variables) == 2, "Expect kernel and bias."
  obj.trainable_variables = [dense.kernel]  # Excludes bias on purpose.
  assert(len(model.losses) == 1), "Expect 1 regularization loss."
  obj.regularization_losses = [
      tf.function(lambda: model.losses[0], input_signature=[])]
  tf.saved_model.save(obj, export_dir)


def _save_batch_norm_model(export_dir):
  """Writes a Hub-style SavedModel."""
  inp = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
  bn = tf.keras.layers.BatchNormalization(momentum=0.8)
  outp = bn(inp)
  model = tf.keras.Model(inp, outp)

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
  assert set(obj.trainable_variables) == set(model.trainable_variables)
  obj.variables = [bn.beta, bn.gamma, bn.moving_mean, bn.moving_variance]
  assert set(obj.variables) == set(
      model.trainable_variables + model.non_trainable_variables)
  obj.regularization_losses = []
  assert set(model.losses) == set()
  tf.saved_model.save(obj, export_dir)


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


class KerasTest(tf.test.TestCase):
  """Tests KerasLayer in an all-Keras environment."""

  def testHalfPlusOneRetraining(self):
    # Import the half-plus-one model into a consumer model.
    export_dir = os.path.join(self.get_temp_dir(), "half-plus-one")
    _save_half_plus_one_model(export_dir)
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
    self.assertEqual(model.trainable_weights[0].shape.rank, 2)  # Kernel.
    self.assertEqual(len(model.non_trainable_weights), 1)
    self.assertEqual(model.non_trainable_weights[0].shape.rank, 1)  # Bias.
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

  def testRegularizationLoss(self):
    # Import the half-plus-one model into a consumer model.
    export_dir = os.path.join(self.get_temp_dir(), "half-plus-one")
    _save_half_plus_one_model(export_dir)
    inp = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
    imported = hub.KerasLayer(export_dir, trainable=False)
    outp = imported(inp)
    model = tf.keras.Model(inp, outp)
    # When untrainable, the layer does not contribute regularization losses.
    self.assertAllEqual(model.losses, np.array([], dtype=np.float32))
    # When trainable (even set after the fact), the layer forwards its losses.
    imported.trainable = True
    self.assertAllEqual(model.losses, np.array([0.0025], dtype=np.float32))
    # This can be toggled repeatedly.
    imported.trainable = False
    self.assertAllEqual(model.losses, np.array([], dtype=np.float32))
    imported.trainable = True
    self.assertAllEqual(model.losses, np.array([0.0025], dtype=np.float32))

  def testBatchNormRetraining(self):
    """Tests imported batch norm with trainable=True."""
    export_dir = os.path.join(self.get_temp_dir(), "batch-norm")
    _save_batch_norm_model(export_dir)
    inp = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
    imported = hub.KerasLayer(export_dir, trainable=True)
    var_beta, var_gamma, var_mean, var_variance = imported.variables
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

  def testBatchNormFreezing(self):
    """Tests imported batch norm with trainable=False."""
    export_dir = os.path.join(self.get_temp_dir(), "batch-norm")
    _save_batch_norm_model(export_dir)
    inp = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
    imported = hub.KerasLayer(export_dir, trainable=False)
    var_beta, var_gamma, var_mean, var_variance = imported.variables
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

  def testComputeOutputShape(self):
    export_dir = os.path.join(self.get_temp_dir(), "half-plus-one")
    _save_half_plus_one_model(export_dir)
    layer = hub.KerasLayer(export_dir, output_shape=[1])
    self.assertEqual([10, 1],
                     layer.compute_output_shape(tuple([10, 1])).as_list())

  def testGetConfigFromConfig(self):
    export_dir = os.path.join(self.get_temp_dir(), "half-plus-one")
    _save_half_plus_one_model(export_dir)
    layer = hub.KerasLayer(export_dir)
    in_value = np.array([[10.0]], dtype=np.float32)
    result = layer(in_value).numpy()

    config = layer.get_config()
    new_layer = hub.KerasLayer.from_config(config)
    new_result = new_layer(in_value).numpy()
    self.assertEqual(result, new_result)

  def testGetConfigFromConfigWithHParams(self):
    if tf.__version__ == "2.0.0-alpha0":
      self.skipTest("b/127938157 broke use of default hparams")
    export_dir = os.path.join(self.get_temp_dir(), "with-hparams")
    _save_model_with_hparams(export_dir)
    layer = hub.KerasLayer(export_dir, arguments=dict(a=10.))  # Leave b=0.
    in_value = np.array([[1.], [2.], [3.]], dtype=np.float32)
    expected_result = np.array([[10.], [20.], [30.]], dtype=np.float32)
    result = layer(in_value).numpy()
    self.assertAllEqual(expected_result, result)

    config = layer.get_config()
    new_layer = hub.KerasLayer.from_config(config)
    new_result = new_layer(in_value).numpy()
    self.assertAllEqual(result, new_result)

  def testSaveModelConfig(self):
    export_dir = os.path.join(self.get_temp_dir(), "half-plus-one")
    _save_half_plus_one_model(export_dir)

    model = tf.keras.Sequential([hub.KerasLayer(export_dir)])
    in_value = np.array([[10.]], dtype=np.float32)
    result = model(in_value).numpy()

    json_string = model.to_json()
    new_model = tf.keras.models.model_from_json(
        json_string, custom_objects={"KerasLayer": hub.KerasLayer})
    new_result = new_model(in_value).numpy()
    self.assertEqual(result, new_result)


class EstimatorTest(tf.test.TestCase):
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

  def testHalfPlusOneRetraining(self):
    export_dir = os.path.join(self.get_temp_dir(), "half-plus-one")
    _save_half_plus_one_model(export_dir)
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

  def testHalfPlusOneFrozen(self):
    export_dir = os.path.join(self.get_temp_dir(), "half-plus-one")
    _save_half_plus_one_model(export_dir)
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
    var_beta, var_gamma, var_mean, var_variance = imported.variables
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

  def testBatchNormRetraining(self):
    """Tests imported batch norm with trainable=True."""
    export_dir = os.path.join(self.get_temp_dir(), "batch-norm")
    _save_batch_norm_model(export_dir)
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

  def testBatchNormFreezing(self):
    """Tests imported batch norm with trainable=False."""
    export_dir = os.path.join(self.get_temp_dir(), "batch-norm")
    _save_batch_norm_model(export_dir)
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


if __name__ == "__main__":
  # The file under test is not imported if TensorFlow is too old.
  # In such an environment, this test should be silently skipped.
  if hasattr(hub, "KerasLayer"):
    # At this point, we are either in in a late TF1 version or in TF2.
    # In TF1, we need to enable V2-like behavior, notably eager execution.
    # `tf.enable_v2_behavior` seems to be missing at times, but
    # `tf.enable_eager_behavior` has been around for long, so we call that.
    # In TF2, those enable_*() methods are unnecessary and no longer available.
    if hasattr(tf, "enable_eager_execution"):
      tf.enable_eager_execution()
    tf.test.main()
