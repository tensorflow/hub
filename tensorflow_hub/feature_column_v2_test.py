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
"""Tests for tensorflow_hub.feature_column."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub


# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.ops.lookup_ops import HashTable
from tensorflow.python.ops.lookup_ops import KeyValueTensorInitializer
# pylint: enable=g-direct-tensorflow-import


class TextEmbedding(tf.train.Checkpoint):

  def __init__(self, returns_dict=False):
    embeddings = [
        ("", [0, 0, 0, 0]),  # OOV items are mapped to this embedding.
        ("hello world", [1, 2, 3, 4]),
        ("pair-programming", [5, 5, 5, 5]),
    ]
    keys = tf.constant([item[0] for item in embeddings], dtype=tf.string)
    indices = tf.constant(list(range(len(embeddings))), dtype=tf.int64)
    tbl_init = KeyValueTensorInitializer(keys, indices)
    self.table = HashTable(tbl_init, 0)
    self.weights = tf.Variable(
        list([item[1] for item in embeddings]), dtype=tf.float32)
    self.variables = [self.weights]
    self.trainable_variables = self.variables
    self._returns_dict = returns_dict

  @tf.function(input_signature=[
      tf.TensorSpec(dtype=tf.string, name="text", shape=[None])
  ])
  def __call__(self, text_tensor):
    indices_tensor = self.table.lookup(text_tensor)
    embedding_tensor = tf.gather(self.weights, indices_tensor)
    return dict(
        outputs=embedding_tensor) if self._returns_dict else embedding_tensor


class TextEmbeddingColumnTest(tf.test.TestCase):

  def setUp(self):
    super(TextEmbeddingColumnTest, self).setUp()
    self.model = os.path.join(self.get_temp_dir(), "model")
    tf.saved_model.save(TextEmbedding(), self.model)
    self.model_returning_dicts = os.path.join(self.get_temp_dir(),
                                              "model_returning_dicts")
    tf.saved_model.save(
        TextEmbedding(returns_dict=True), self.model_returning_dicts)

  def testParents(self):
    text_column = hub.text_embedding_column_v2(
        "text", self.model, trainable=False)
    self.assertEqual(["text"], text_column.parents)

  def testMakeParseExampleSpec(self):
    text_column = hub.text_embedding_column_v2(
        "text", self.model, trainable=False)
    parsing_spec = tf.feature_column.make_parse_example_spec([text_column])
    self.assertEqual(parsing_spec,
                     {"text": tf.io.FixedLenFeature([1], dtype=tf.string)})

  def testFeatureColumnsIsV2(self):
    feature_column = hub.text_embedding_column_v2("text_a", self.model)
    self.assertTrue(feature_column_v2.is_feature_column_v2([feature_column]))

  def testConfig(self):
    text_column = hub.text_embedding_column_v2(
        "text", self.model, trainable=True)
    config = text_column.get_config()
    cloned_column = hub.feature_column_v2._TextEmbeddingColumnV2.from_config(
        config)
    self.assertEqual(cloned_column.module_path, text_column.module_path)

  def testDenseFeaturesDirectly(self):
    features = {
        "text_a": ["hello world", "pair-programming"],
        "text_b": ["hello world", "oov token"],
    }
    feature_columns = [
        hub.text_embedding_column_v2("text_a", self.model, trainable=False),
        hub.text_embedding_column_v2("text_b", self.model, trainable=False),
    ]
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    feature_layer_out = feature_layer(features)
    self.assertAllEqual(feature_layer_out,
                        [[1, 2, 3, 4, 1, 2, 3, 4], [5, 5, 5, 5, 0, 0, 0, 0]])

  def testDenseFeaturesInKeras(self):
    features = {
        "text": np.array(["hello world", "pair-programming"]),
    }
    label = np.int64([0, 1])
    feature_columns = [
        hub.text_embedding_column_v2("text", self.model, trainable=True),
    ]
    input_features = dict(
        text=tf.keras.layers.Input(name="text", shape=[None], dtype=tf.string))
    dense_features = tf.keras.layers.DenseFeatures(feature_columns)
    x = dense_features(input_features)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    logits = tf.keras.layers.Dense(1, activation="linear")(x)
    model = tf.keras.Model(inputs=input_features, outputs=logits)
    model.compile(
        optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x=features, y=label, epochs=10)
    self.assertAllEqual(model.predict(features["text"]).shape, [2, 1])

  def testLoadingDifferentFeatureColumnsFails(self):
    features = [
        np.array(["hello world", "pair-programming"]),
        np.array(["hello world", "pair-programming"]),
    ]
    label = np.int64([0, 1])
    feature_columns = [
        hub.text_embedding_column_v2("text_1", self.model, trainable=True),
    ]
    # Build the first model.
    input_features = dict(
        text_1=tf.keras.layers.Input(
            name="text_1", shape=[None], dtype=tf.string))
    dense_features = tf.keras.layers.DenseFeatures(feature_columns)
    x = dense_features(input_features)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    logits = tf.keras.layers.Dense(1, activation="linear")(x)
    model_1 = tf.keras.Model(inputs=input_features, outputs=logits)
    model_1.compile(
        optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    model_1.fit(x=features, y=label, epochs=10)

    checkpoint_path = os.path.join(self.get_temp_dir(), "checkpoints",
                                   "checkpoint-1")
    model_1.save_weights(checkpoint_path)

    # Build the second model with feature columns that have different names.
    feature_columns = [
        hub.text_embedding_column_v2("text_2", self.model, trainable=True),
    ]
    input_features = dict(
        text_2=tf.keras.layers.Input(
            name="text_2", shape=[None], dtype=tf.string))
    dense_features = tf.keras.layers.DenseFeatures(feature_columns)
    x = dense_features(input_features)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    logits = tf.keras.layers.Dense(1, activation="linear")(x)
    model_2 = tf.keras.Model(inputs=input_features, outputs=logits)
    model_2.compile(
        optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

    # Loading of checkpoints from the first model into the second model should
    # fail.
    with self.assertRaisesRegexp(AssertionError,
                                 ".*Some Python objects were not bound.*"):
      model_2.load_weights(checkpoint_path).assert_consumed()

  def testWorksWithTF2DnnClassifier(self):
    comment_embedding_column = hub.text_embedding_column_v2(
        "comment", self.model, trainable=False)
    upvotes = tf.feature_column.numeric_column("upvotes")

    feature_columns = [comment_embedding_column, upvotes]
    estimator = tf.estimator.DNNClassifier(
        hidden_units=[10],
        feature_columns=feature_columns,
        model_dir=self.get_temp_dir())

    # This only tests that estimator apis are working with the feature
    # column without throwing exceptions.
    def input_fn():
      features = {
          "comment": np.array([
              ["the quick brown fox"],
              ["spam spam spam"],
          ]),
          "upvotes": np.array([
              [20],
              [1],
          ]),
      }
      labels = np.array([[1], [0]])
      return features, labels
    estimator.train(input_fn, max_steps=1)
    estimator.evaluate(input_fn, steps=1)
    estimator.predict(input_fn)

  def testWorksWithDNNEstimatorAndDataset(self):
    self.skipTest("b/154115879 - needs more investigation for timeout.")
    description_embeddings = hub.text_embedding_column_v2(
        "descriptions", self.model_returning_dicts, output_key="outputs")

    def input_fn():
      features = dict(descriptions=tf.constant([["sentence"]]))
      labels = tf.constant([[1]])
      dataset = tf.data.Dataset.from_tensor_slices((features, labels))

      data_batches = dataset.repeat().take(30).batch(5)
      return data_batches

    estimator = tf.estimator.DNNEstimator(
        model_dir=os.path.join(self.get_temp_dir(), "estimator_export"),
        hidden_units=[10],
        head=tf.estimator.BinaryClassHead(),
        feature_columns=[description_embeddings])

    estimator.train(input_fn=input_fn, max_steps=1)


if __name__ == "__main__":
  # This test is only supported in TF2 mode and only in TensorFlow version that
  # has the following symbol:
  # tensorflow.python.feature_column.feature_column_v2.StateManager.has_resource
  if tf.executing_eagerly() and hasattr(feature_column_v2.StateManager,
                                        "has_resource"):
    logging.info("Using TF version: %s", tf.__version__)
    tf.test.main()
  else:
    logging.warning("Skipping running tests for TF Version: %s", tf.__version__)
