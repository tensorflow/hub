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
r"""Tests for MNIST exporter."""

import logging
import os

import tensorflow as tf
import tensorflow_hub as hub

from examples.mnist_export_v2 import export


class ExportTest(tf.test.TestCase):
  """Test for MNIST model exporter."""

  def setUp(self):
    super().setUp()
    def create_image_and_label(index):
      image = tf.image.convert_image_dtype(
          255 * tf.random.normal([1, 28, 28, 1]), dtype=tf.uint8, saturate=True)
      return dict(image=image, label=[index])
    self.mock_dataset = tf.data.Dataset.range(5).map(create_image_and_label,num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def test_model_exporting(self):
    export.train_and_export(
        epoch=1,
        dataset=self.mock_dataset,
        export_path="%s/model/1" % self.get_temp_dir())
    self.assertTrue(os.listdir(self.get_temp_dir()))

  def test_empty_input(self):
    export.train_and_export(
        epoch=1,
        dataset=self.mock_dataset,
        export_path="%s/model/1" % self.get_temp_dir())
    model = hub.load("%s/model/1" % self.get_temp_dir())
    output_ = model(tf.zeros([1, 28, 28, 1], dtype=tf.uint8).numpy())
    self.assertEqual(output_.shape, [1, 10])


if __name__ == "__main__":
  # This test is only supported in TF 2.0.
  if tf.executing_eagerly():
    logging.info("Using TF version: %s", tf.__version__)
    tf.test.main()
  else:
    logging.warning("Skipping running tests for TF Version: %s", tf.__version__)
