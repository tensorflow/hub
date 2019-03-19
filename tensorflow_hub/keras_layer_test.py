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


class KerasLayerTest(tf.test.TestCase):

  def testHalfPlusOneExample(self):
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
