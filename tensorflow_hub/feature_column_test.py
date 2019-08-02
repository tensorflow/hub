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
"""Tests for tensorflow_hub.feature_column."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint:disable=g-import-not-at-top,g-statement-before-imports
try:
  import mock as mock
except ImportError:
  import unittest.mock as mock
# pylint:disable=g-import-not-at-top,g-statement-before-imports

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_hub import test_utils
from tensorflow_hub import tf_v1

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.ops.lookup_ops import HashTable
from tensorflow.python.ops.lookup_ops import KeyValueTensorInitializer
# pylint: enable=g-direct-tensorflow-import

_dense_features_module = test_utils.get_dense_features_module()


def text_module_fn():
  embeddings = [
      ("", [0, 0, 0, 0]),  # OOV items are mapped to this embedding.
      ("hello world", [1, 2, 3, 4]),
      ("pair-programming", [5, 5, 5, 5]),
  ]
  keys = tf.constant([item[0] for item in embeddings], dtype=tf.string)
  indices = tf.constant(list(range(len(embeddings))), dtype=tf.int64)
  tbl_init = KeyValueTensorInitializer(keys, indices)
  table = HashTable(tbl_init, 0)

  weights_initializer = tf.cast(
      tf.constant(list([item[1] for item in embeddings])), tf.float32)

  weights = tf_v1.get_variable(
      "weights", dtype=tf.float32, initializer=weights_initializer)

  text_tensor = tf_v1.placeholder(dtype=tf.string, name="text", shape=[None])
  indices_tensor = table.lookup(text_tensor)
  embedding_tensor = tf.gather(weights, indices_tensor)
  hub.add_signature(inputs=text_tensor, outputs=embedding_tensor)


def invalid_text_module_fn():
  text = tf_v1.placeholder(tf.string, shape=[10])
  hub.add_signature(inputs=text, outputs=tf.zeros([10, 3]))


class CommonColumnTest(tf.test.TestCase):

  def setUp(self):
    self.spec = hub.create_module_spec(text_module_fn)

  @mock.patch.object(feature_column_v2._StateManagerImpl, "add_resource")
  def testFeatureColumnsWithResources(self, mock_add_resource):
    feature_column = hub.text_embedding_column("text_a", self.spec)
    if not isinstance(feature_column, feature_column_v2.FeatureColumn):
      self.skipTest("Resources not implemented in the state manager of feature "
                    "column v2.")
    self.assertTrue(feature_column_v2.is_feature_column_v2([feature_column]))

  @mock.patch.object(feature_column_v2._StateManagerImpl, "add_resource")
  def testFeatureColumnsWithNoResources(self, mock_add_resource):
    mock_add_resource.side_effect = NotImplementedError
    feature_column = hub.text_embedding_column("text_a", self.spec)
    self.assertFalse(feature_column_v2.is_feature_column_v2([feature_column]))


class TextEmbeddingColumnTest(tf.test.TestCase):

  def setUp(self):
    self.spec = hub.create_module_spec(text_module_fn)

  def testVariableShape(self):
    text_column = hub.text_embedding_column("text", self.spec, trainable=False)
    self.assertEqual(text_column._variable_shape, [4])

  def testParents(self):
    text_column = hub.text_embedding_column("text", self.spec, trainable=False)
    self.assertEqual(["text"], text_column.parents)

  def testMakeParseExampleSpec(self):
    text_column = hub.text_embedding_column("text", self.spec, trainable=False)
    parsing_spec = tf_v1.feature_column.make_parse_example_spec([text_column])
    self.assertEqual(parsing_spec,
                     {"text": tf_v1.FixedLenFeature([1], dtype=tf.string)})

  def testInputLayer(self):
    features = {
        "text_a": ["hello world", "pair-programming"],
        "text_b": ["hello world", "oov token"],
    }
    feature_columns = [
        hub.text_embedding_column("text_a", self.spec, trainable=False),
        hub.text_embedding_column("text_b", self.spec, trainable=False),
    ]
    with tf.Graph().as_default():
      input_layer = tf_v1.feature_column.input_layer(features, feature_columns)
      with tf_v1.train.MonitoredSession() as sess:
        output = sess.run(input_layer)
        self.assertAllEqual(
            output, [[1, 2, 3, 4, 1, 2, 3, 4], [5, 5, 5, 5, 0, 0, 0, 0]])

  def testDenseFeatures(self):
    features = {
        "text_a": ["hello world", "pair-programming"],
        "text_b": ["hello world", "oov token"],
    }
    feature_columns = [
        hub.text_embedding_column("text_a", self.spec, trainable=False),
        hub.text_embedding_column("text_b", self.spec, trainable=False),
    ]
    if not feature_column_v2.is_feature_column_v2(feature_columns):
      self.skipTest("Resources not implemented in the state manager of feature "
                    "column v2.")
    with tf.Graph().as_default():
      feature_layer = _dense_features_module.DenseFeatures(feature_columns)
      feature_layer_out = feature_layer(features)
      with tf_v1.train.MonitoredSession() as sess:
        output = sess.run(feature_layer_out)
        self.assertAllEqual(
            output, [[1, 2, 3, 4, 1, 2, 3, 4], [5, 5, 5, 5, 0, 0, 0, 0]])

  def testDenseFeatures_shareAcrossApplication(self):
    features = {
        "text": ["hello world", "pair-programming"],
    }
    feature_columns = [
        hub.text_embedding_column("text", self.spec, trainable=True),
    ]
    if not feature_column_v2.is_feature_column_v2(feature_columns):
      self.skipTest("Resources not implemented in the state manager of feature "
                    "column v2.")
    with tf.Graph().as_default():
      feature_layer = _dense_features_module.DenseFeatures(feature_columns)
      feature_layer_out_1 = feature_layer(features)
      feature_layer_out_2 = feature_layer(features)

      # We define loss only on the first layer. Since layers should have shared
      # weights, we expect the second layer will change too.
      loss = feature_layer_out_1 - tf.constant(0.005)
      optimizer = tf_v1.train.GradientDescentOptimizer(learning_rate=0.7)
      train_op = optimizer.minimize(loss)

      with tf_v1.train.MonitoredSession() as sess:
        before_update_1 = sess.run(feature_layer_out_1)
        sess.run(train_op)
        after_update_1 = sess.run(feature_layer_out_1)
        after_update_2 = sess.run(feature_layer_out_2)

        self.assertAllEqual(before_update_1, [[1, 2, 3, 4],
                                              [5, 5, 5, 5]])
        self.assertAllEqual(after_update_1, after_update_2)

  def testWorksWithCannedEstimator(self):
    comment_embedding_column = hub.text_embedding_column(
        "comment", self.spec, trainable=False)
    upvotes = tf_v1.feature_column.numeric_column("upvotes")

    feature_columns = [comment_embedding_column, upvotes]
    estimator = tf_v1.estimator.DNNClassifier(
        hidden_units=[10],
        feature_columns=feature_columns,
        model_dir=self.get_temp_dir())

    # This only tests that estimator apis are working with the feature
    # column without throwing exceptions.
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
    if hasattr(tf.compat, "v1"):
      numpy_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn
    else:
      numpy_input_fn = tf_v1.estimator.inputs.numpy_input_fn
    input_fn = numpy_input_fn(features, labels, shuffle=True)
    estimator.train(input_fn, max_steps=1)
    estimator.evaluate(input_fn, steps=1)
    estimator.predict(input_fn)

  def testTrainableEmbeddingColumn(self):
    feature_columns = [
        hub.text_embedding_column("text", self.spec, trainable=True),
    ]

    with tf.Graph().as_default():
      features = {
          "text": ["hello world", "pair-programming"],
      }
      target = [[1, 1, 1, 1], [4, 3, 2, 1]]
      input_layer = tf_v1.feature_column.input_layer(features, feature_columns)

      loss = tf.cast(
          tf_v1.losses.mean_squared_error(input_layer, target), tf.float64)
      optimizer = tf_v1.train.GradientDescentOptimizer(learning_rate=0.97)
      train_op = optimizer.minimize(loss)

      with tf_v1.train.MonitoredSession() as sess:
        self.assertAllEqual(sess.run(input_layer), [[1, 2, 3, 4], [5, 5, 5, 5]])
        for _ in range(10):
          sess.run(train_op)
        self.assertAllClose(sess.run(input_layer), target, atol=0.5)

  def testInvalidTextModule(self):
    spec = hub.create_module_spec(invalid_text_module_fn)
    with self.assertRaisesRegexp(ValueError, "only one input"):
      hub.text_embedding_column("coment", spec, trainable=False)


def create_image_module_fn(randomly_initialized=False):
  def image_module_fn():
    """Maps 1x2 images to sums of each color channel."""
    images = tf_v1.placeholder(dtype=tf.float32, shape=[None, 1, 2, 3])
    if randomly_initialized:
      initializer = tf_v1.random_uniform_initializer(
          minval=-1, maxval=1, dtype=tf.float32)
    else:
      initializer = tf_v1.constant_initializer(1.0, dtype=tf.float32)
    weight = tf_v1.get_variable(
        name="weight", shape=[1], initializer=initializer)
    sum_channels = tf.reduce_sum(images, axis=[1, 2]) * weight
    hub.add_signature(inputs={"images": images}, outputs=sum_channels)
  return image_module_fn


class ImageEmbeddingColumnTest(tf.test.TestCase):

  def setUp(self):
    self.spec = hub.create_module_spec(create_image_module_fn())
    self.randomly_initialized_spec = hub.create_module_spec(
        create_image_module_fn(randomly_initialized=True))

  def testExpectedImageSize(self):
    image_column = hub.image_embedding_column("image", self.spec)
    # The usage comment recommends this code pattern, so we test it here.
    self.assertSequenceEqual(
        hub.get_expected_image_size(image_column.module_spec), [1, 2])

  def testVariableShape(self):
    image_column = hub.image_embedding_column("image", self.spec)
    self.assertEqual(image_column.variable_shape, [3])

  def testParents(self):
    image_column = hub.image_embedding_column("image", self.spec)
    self.assertEqual(["image"], image_column.parents)

  def testMakeParseExampleSpec(self):
    image_column = hub.image_embedding_column("image", self.spec)
    parsing_spec = tf_v1.feature_column.make_parse_example_spec([image_column])
    self.assertEqual(
        parsing_spec,
        {"image": tf_v1.FixedLenFeature([1, 2, 3], dtype=tf.float32)})

  def testInputLayer(self):
    features = {
        "image_a": [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
                    [[[0.7, 0.7, 0.7], [0.1, 0.2, 0.3]]]],
        "image_b": [[[[0.1, 0.2, 0.1], [0.2, 0.1, 0.2]]],
                    [[[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]]],
    }
    feature_columns = [
        hub.image_embedding_column("image_a", self.spec),
        hub.image_embedding_column("image_b", self.spec),
    ]
    with tf.Graph().as_default():
      input_layer = tf_v1.feature_column.input_layer(features, feature_columns)
      with tf_v1.train.MonitoredSession() as sess:
        output = sess.run(input_layer)
        self.assertAllClose(
            output,
            [[0.5, 0.7, 0.9, 0.3, 0.3, 0.3], [0.8, 0.9, 1.0, 0.4, 0.4, 0.4]])

  def testDenseFeatures(self):
    features = {
        "image_a": [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
                    [[[0.7, 0.7, 0.7], [0.1, 0.2, 0.3]]]],
        "image_b": [[[[0.1, 0.2, 0.1], [0.2, 0.1, 0.2]]],
                    [[[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]]],
    }
    feature_columns = [
        hub.image_embedding_column("image_a", self.spec),
        hub.image_embedding_column("image_b", self.spec),
    ]
    if not feature_column_v2.is_feature_column_v2(feature_columns):
      self.skipTest("Resources not implemented in the state manager of feature "
                    "column v2.")
    with tf.Graph().as_default():
      feature_layer = _dense_features_module.DenseFeatures(feature_columns)
      feature_layer_out = feature_layer(features)
      with tf_v1.train.MonitoredSession() as sess:
        output = sess.run(feature_layer_out)
        self.assertAllClose(
            output,
            [[0.5, 0.7, 0.9, 0.3, 0.3, 0.3], [0.8, 0.9, 1.0, 0.4, 0.4, 0.4]])

  def testDenseFeatures_shareAcrossApplication(self):
    features = {
        "image": [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
                  [[[0.7, 0.7, 0.7], [0.1, 0.2, 0.3]]]],
    }
    feature_columns = [
        hub.image_embedding_column("image", self.randomly_initialized_spec),
    ]
    if not feature_column_v2.is_feature_column_v2(feature_columns):
      self.skipTest("Resources not implemented in the state manager of feature "
                    "column v2.")
    with tf.Graph().as_default():
      feature_layer = _dense_features_module.DenseFeatures(feature_columns)
      feature_layer_out_1 = feature_layer(features)
      feature_layer_out_2 = feature_layer(features)

      with tf_v1.train.MonitoredSession() as sess:
        output_1 = sess.run(feature_layer_out_1)
        output_2 = sess.run(feature_layer_out_2)

        self.assertAllClose(output_1, output_2)

  def testWorksWithCannedEstimator(self):
    image_column = hub.image_embedding_column("image", self.spec)
    other_column = tf_v1.feature_column.numeric_column("number")

    feature_columns = [image_column, other_column]
    estimator = tf_v1.estimator.DNNClassifier(
        hidden_units=[10],
        feature_columns=feature_columns,
        model_dir=self.get_temp_dir())

    # This only tests that estimator apis are working with the feature
    # column without throwing exceptions.
    features = {
        "image":
            np.array([[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
                      [[[0.7, 0.7, 0.7], [0.1, 0.2, 0.3]]]],
                     dtype=np.float32),
        "number":
            np.array([[20], [1]]),
    }
    labels = np.array([[1], [0]])
    if hasattr(tf.compat, "v1"):
      numpy_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn
    else:
      numpy_input_fn = tf_v1.estimator.inputs.numpy_input_fn
    input_fn = numpy_input_fn(features, labels, shuffle=True)
    estimator.train(input_fn, max_steps=1)
    estimator.evaluate(input_fn, steps=1)
    estimator.predict(input_fn)


class SparseTextEmbeddingColumnTest(tf.test.TestCase):

  def setUp(self):
    self.spec = hub.create_module_spec(text_module_fn)

  def testVariableShape(self):
    text_column = hub.sparse_text_embedding_column(
        "text", self.spec, combiner="mean", default_value=None, trainable=False)
    self.assertEqual(text_column._variable_shape, [4])

  def testMakeParseExampleSpec(self):
    text_column = hub.sparse_text_embedding_column(
        "text", self.spec, combiner="mean", default_value=None, trainable=False)
    parsing_spec = tf_v1.feature_column.make_parse_example_spec([text_column])
    self.assertEqual(parsing_spec, {"text": tf_v1.VarLenFeature(tf.string)})

  def testInputLayer(self):
    with tf.Graph().as_default():
      text_a = tf.SparseTensor(
          values=["hello world", "pair-programming", "hello world"],
          indices=[[0, 0], [0, 1], [1, 0]],
          dense_shape=[2, 2])
      text_b = tf.SparseTensor(
          values=["hello world", "oov token"],
          indices=[[0, 0], [0, 1]],
          dense_shape=[2, 3])

      features = {
          "text_a": text_a,
          "text_b": text_b,
      }
      feature_columns = [
          hub.sparse_text_embedding_column(
              "text_a",
              self.spec,
              combiner="mean",
              default_value="__UNKNOWN__",
              trainable=False),
          hub.sparse_text_embedding_column(
              "text_b",
              self.spec,
              combiner="mean",
              default_value="__UNKNOWN__",
              trainable=False),
      ]
      input_layer = tf_v1.feature_column.input_layer(features, feature_columns)
      with tf_v1.train.MonitoredSession() as sess:
        output = sess.run(input_layer)
        self.assertAllEqual(
            output,
            [[3, 3.5, 4, 4.5, 0.5, 1, 1.5, 2], [1, 2, 3, 4, 0, 0, 0, 0]])
        # ([1, 2, 3, 4] + [5, 5, 5, 5])/2 extend ([1, 2, 3, 4] + [0, 0, 0, 0])/2
        # [1, 2, 3, 4] extend [0, 0, 0, 0]

  def testTrainableEmbeddingColumn(self):
    feature_columns = [
        hub.sparse_text_embedding_column(
            "text",
            self.spec,
            combiner="mean",
            default_value=None,
            trainable=True),
    ]

    with tf.Graph().as_default():
      text = tf.SparseTensor(
          values=["hello world", "pair-programming"],
          indices=[[0, 0], [1, 0]],
          dense_shape=[2, 2])

      target = [[1, 1, 1, 1], [4, 3, 2, 1]]
      input_layer = tf_v1.feature_column.input_layer({"text": text},
                                                     feature_columns)

      loss = tf_v1.losses.mean_squared_error(input_layer, target)
      optimizer = tf_v1.train.GradientDescentOptimizer(learning_rate=0.97)
      train_op = optimizer.minimize(loss)

      with tf_v1.train.MonitoredSession() as sess:
        self.assertAllEqual(sess.run(input_layer), [[1, 2, 3, 4], [5, 5, 5, 5]])
        for _ in range(10):
          sess.run(train_op)
        self.assertAllClose(sess.run(input_layer), target, atol=0.5)

  def testEmptySparseTensorBatch(self):
    feature_columns = [
        hub.sparse_text_embedding_column(
            "text",
            self.spec,
            combiner="mean",
            default_value="default",
            trainable=True),
    ]

    with tf.Graph().as_default():
      text = tf.SparseTensor(
          values=tf_v1.constant([], dtype=tf_v1.string, shape=[0]),
          indices=tf_v1.constant([], dtype=tf_v1.int64, shape=[0, 2]),
          dense_shape=[3, 0])

      input_layer = tf_v1.feature_column.input_layer({"text": text},
                                                     feature_columns)

      with tf_v1.train.MonitoredSession() as sess:
        embeddings = sess.run(input_layer)
        self.assertAllEqual(embeddings,
                            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

  def testEmptySparseTensorRow(self):
    feature_columns = [
        hub.sparse_text_embedding_column(
            "text",
            self.spec,
            combiner="mean",
            default_value="default",
            trainable=True),
    ]

    with tf.Graph().as_default():
      text = tf.SparseTensor(
          values=tf_v1.constant(["hello world"], dtype=tf_v1.string, shape=[1]),
          indices=tf_v1.constant([[0, 0]], dtype=tf_v1.int64, shape=[1, 2]),
          dense_shape=[2, 1])

      input_layer = tf_v1.feature_column.input_layer({"text": text},
                                                     feature_columns)

      with tf_v1.train.MonitoredSession() as sess:
        embeddings = sess.run(input_layer)
        self.assertAllEqual(embeddings, [[1, 2, 3, 4], [0, 0, 0, 0]])


if __name__ == "__main__":
  tf.test.main()
