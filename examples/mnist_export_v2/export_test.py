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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
from distutils.version import LooseVersion

import tensorflow as tf
import tensorflow_hub as hub

from examples.mnist_export_v2 import export


class TFHubMNISTTest(tf.test.TestCase):

  def setUp(self):
    self.mock_dataset = tf.data.Dataset.range(5).map(
        lambda x: {
            "image": tf.cast(255 * tf.random.normal([1, 28, 28, 1]), tf.uint8),
            "label": x
        })

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
    output_ = model.call(tf.zeros([1, 28, 28, 1], dtype=tf.uint8).numpy())
    self.assertEqual(output_.shape, [1, 10])


if __name__ == "__main__":
  # This test is only supported in TF 2.0.
  if LooseVersion(tf.__version__) >= LooseVersion("2.0.0-beta0"):
    logging.info("Using TF version: %s", tf.__version__)
    tf.test.main()
  else:
    logging.warning("Skipping running tests for TF Version: %s", tf.__version__)
