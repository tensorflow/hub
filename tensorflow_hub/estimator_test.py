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
"""Tests for tensorflow_hub.estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import tensorflow as tf
import tensorflow_hub as hub

_TEXT_FEATURE_NAME = "text"
_EXPORT_MODULE_NAME = "embedding-text"


def _input_fn():
  """An input fn."""
  features = {
      _TEXT_FEATURE_NAME: tf.constant([
          "Example 1 feature", "Example 2"]),
  }
  labels = tf.constant([False, True])
  return features, labels


def _serving_input_fn():
  """A serving input fn."""
  text_features = tf.placeholder(dtype=tf.string, shape=[None])
  return tf.estimator.export.ServingInputReceiver(
      features={_TEXT_FEATURE_NAME: text_features},
      receiver_tensors=text_features)


def text_module_fn():
  weights = tf.get_variable(
      "weights", dtype=tf.float32, shape=[100, 10])
  #      initializer=tf.random_uniform_initializer())
  text = tf.placeholder(tf.string, shape=[None])
  hash_buckets = tf.string_to_hash_bucket_fast(text, weights.get_shape()[0])
  embeddings = tf.gather(weights, hash_buckets)
  hub.add_signature(inputs=text, outputs=embeddings)


def _get_model_fn(register_module=False):
  def _model_fn(features, labels, mode):
    """A model_fn that uses a mock TF-Hub module."""
    del labels

    spec = hub.create_module_spec(text_module_fn)
    embedding = hub.Module(spec)
    if register_module:
      hub.register_module_for_export(embedding, _EXPORT_MODULE_NAME)
    predictions = embedding(features[_TEXT_FEATURE_NAME])
    loss = tf.constant(0.0)

    global_step = tf.train.get_global_step()
    train_op = tf.assign_add(global_step, 1)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op)

  return _model_fn


class EstimatorTest(tf.test.TestCase):

  def testLatestModuleExporterDirectly(self):
    model_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    export_base_dir = os.path.join(
        tempfile.mkdtemp(dir=self.get_temp_dir()), "export")

    estimator = tf.estimator.Estimator(_get_model_fn(register_module=True),
                                       model_dir=model_dir)
    estimator.train(input_fn=_input_fn, steps=1)

    exporter = hub.LatestModuleExporter("exporter_name", _serving_input_fn)
    export_dir = exporter.export(estimator=estimator,
                                 export_path=export_base_dir,
                                 eval_result=None,
                                 is_the_final_export=None)

    # Check that a timestamped directory is created in the expected location.
    timestamp_dirs = tf.gfile.ListDirectory(export_base_dir)
    self.assertEquals(1, len(timestamp_dirs))
    self.assertEquals(
        tf.compat.as_bytes(os.path.join(export_base_dir, timestamp_dirs[0])),
        tf.compat.as_bytes(export_dir))

    # Check the timestamped directory containts the exported modules inside.
    expected_module_dir = os.path.join(
        tf.compat.as_bytes(export_dir),
        tf.compat.as_bytes(_EXPORT_MODULE_NAME))
    self.assertTrue(tf.gfile.IsDirectory(expected_module_dir))

  def test_latest_module_exporter_with_no_modules(self):
    model_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    export_base_dir = os.path.join(tempfile.mkdtemp(dir=self.get_temp_dir()),
                                   "export")
    self.assertFalse(tf.gfile.Exists(export_base_dir))

    estimator = tf.estimator.Estimator(_get_model_fn(register_module=False),
                                       model_dir=model_dir)
    estimator.train(input_fn=_input_fn, steps=1)

    exporter = hub.LatestModuleExporter("exporter_name", _serving_input_fn)
    export_dir = exporter.export(estimator=estimator,
                                 export_path=export_base_dir,
                                 eval_result=None,
                                 is_the_final_export=None)

    # Check the result.
    self.assertIsNone(export_dir)

    # Check that a no directory has been created in the expected location.
    self.assertFalse(tf.gfile.Exists(export_base_dir))

  def test_latest_module_exporter_with_eval_spec(self):
    model_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    estimator = tf.estimator.Estimator(_get_model_fn(register_module=True),
                                       model_dir=model_dir)
    exporter = hub.LatestModuleExporter(
        "tf_hub", _serving_input_fn, exports_to_keep=2)
    estimator.train(_input_fn, max_steps=1)
    export_base_dir = os.path.join(model_dir, "export", "tf_hub")

    exporter.export(estimator, export_base_dir)
    timestamp_dirs = tf.gfile.ListDirectory(export_base_dir)
    self.assertEquals(1, len(timestamp_dirs))
    oldest_timestamp = timestamp_dirs[0]

    expected_module_dir = os.path.join(export_base_dir,
                                       timestamp_dirs[0],
                                       _EXPORT_MODULE_NAME)
    self.assertTrue(tf.gfile.IsDirectory(expected_module_dir))

    exporter.export(estimator, export_base_dir)
    timestamp_dirs = tf.gfile.ListDirectory(export_base_dir)
    self.assertEquals(2, len(timestamp_dirs))

    # Triggering yet another export should clean the oldest export.
    exporter.export(estimator, export_base_dir)
    timestamp_dirs = tf.gfile.ListDirectory(export_base_dir)
    self.assertEquals(2, len(timestamp_dirs))
    self.assertFalse(oldest_timestamp in timestamp_dirs)


if __name__ == "__main__":
  tf.test.main()
