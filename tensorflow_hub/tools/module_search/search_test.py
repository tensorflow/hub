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
"""Tests for module search utility."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint:disable=g-import-not-at-top,g-statement-before-imports
try:
  import mock
except ImportError:
  from unittest import mock
# pylint:disable=g-import-not-at-top,g-statement-before-imports

import os
import numpy as np

from absl.testing import flagsaver
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from tensorflow_hub.tools.module_search import search


class ImageChannelMeanModel(tf.train.Checkpoint):
  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
  ])
  def __call__(self, images):
    return tf.math.reduce_mean(images, [1, 2])


def fake_image_dataset(*args, **kwargs):
  num_examples = 30
  return tf.data.Dataset.from_generator(
      lambda: ({
          "image": np.ones(shape=(32, 32, 3), dtype=np.uint8),
          "label": i % 10,
      } for i in range(num_examples)),
      output_types={"image": tf.uint8, "label": tf.int64},
      output_shapes={"image": (32, 32, 3), "label": ()},
  )


class SearchTest(tf.test.TestCase):

  def _create_image_models(self):
    path1 = os.path.join(self.get_temp_dir(), "model1")
    path2 = os.path.join(self.get_temp_dir(), "model2")
    tf.saved_model.save(ImageChannelMeanModel(), path1)
    tf.saved_model.save(ImageChannelMeanModel(), path2)
    return [path1, path2]

  @mock.patch.object(search.utils.tfds, "load", side_effect=fake_image_dataset)
  def test_run_e2e(self, mock_tfds_load):
    if not tf.executing_eagerly():
      self.skipTest("Test requires eager mode.")
    modules = self._create_image_models()
    #tfds.load = fake_image_dataset
    with flagsaver.flagsaver(
        dataset="cifar100",
        module=modules,
    ):
      search.main([])


if __name__ == '__main__':
  tf.test.main()
