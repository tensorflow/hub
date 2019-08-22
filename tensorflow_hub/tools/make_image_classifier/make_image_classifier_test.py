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
"""Unit tests for make_image_classifier.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import sys

from absl import logging
from absl.testing import flagsaver
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow_hub.tools.make_image_classifier import make_image_classifier


def _write_filled_jpeg_file(path, rgb, image_size):
  r, g, b = rgb
  img = np.broadcast_to(np.array([[[r, g, b]]], dtype=np.uint8),
                        shape=(image_size, image_size, 3))
  tf.keras.preprocessing.image.save_img(path, img, "channels_last", "jpeg")


class MakeImageClassifierTest(tf.test.TestCase):
  IMAGE_SIZE = 24
  IMAGES_PER_CLASS = 20
  DEFAULT_FLAGS = dict(image_size=IMAGE_SIZE, train_epochs=10,
                       batch_size=8, learning_rate=0.1, momentum=0.0)

  def _write_cmy_dataset(self):
    path = os.path.join(self.get_temp_dir(), "cmy_image_dir")
    os.mkdir(path)  # Fails if exists.
    for class_name, rgb in (("cyan", (0, 255, 255)),
                            ("magenta", (255, 0, 255)),
                            ("yellow", (255, 255, 0))):
      class_subdir = os.path.join(path, class_name)
      os.mkdir(class_subdir)
      for i in range(self.IMAGES_PER_CLASS):
        _write_filled_jpeg_file(
            os.path.join(class_subdir, "img_%s_%03d.jpeg" % (class_name, i)),
            rgb, self.IMAGE_SIZE)
    return path

  def _write_random_dataset(self):
    path = os.path.join(self.get_temp_dir(), "random_image_dir")
    os.mkdir(path)  # Fails if exists.
    for class_name in ("ami", "baz", "zrh"):
      class_subdir = os.path.join(path, class_name)
      os.mkdir(class_subdir)
      for i in range(self.IMAGES_PER_CLASS):
        _write_filled_jpeg_file(
            os.path.join(class_subdir, "img_%s_%03d.jpeg" % (class_name, i)),
            [random.uniform(0, 255) for _ in range(3)],
            self.IMAGE_SIZE)
    return path

  def _export_global_average_model(self, has_fixed_input_size=True):
    if has_fixed_input_size:
      input_size = (self.IMAGE_SIZE, self.IMAGE_SIZE, 3)
      dirname = "global_average_fixed_size"
    else:
      input_size = (None, None, 3)
      dirname = "global_average_variable_size"
    path = os.path.join(self.get_temp_dir(), dirname)
    inputs = tf.keras.Input(input_size)
    outputs = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    model = tf.keras.Model(inputs, outputs)
    model.build((None,) + input_size)
    model.save(path, save_format="tf")
    return path

  def testEndToEndSuccess(self):
    logging.info("Using testdata in %s", self.get_temp_dir())
    avg_model_dir = self._export_global_average_model()
    image_dir = self._write_cmy_dataset()
    saved_model_dir = os.path.join(self.get_temp_dir(), "final_saved_model")
    self.assertFalse(  # Make sure we don't test for pre-existing files.
        os.path.isfile(os.path.join(saved_model_dir, "saved_model.pb")))

    with flagsaver.flagsaver(
        image_dir=image_dir, tfhub_module=avg_model_dir,
        # This dataset is expected to be fit perfectly.
        assert_accuracy_at_least=0.9,
        saved_model_dir=saved_model_dir, **self.DEFAULT_FLAGS):
      make_image_classifier.main([])

    # Test for main output artifact.
    self.assertTrue(
        os.path.isfile(os.path.join(saved_model_dir, "saved_model.pb")))

  def testEndToEndAccuracyFailure(self):
    logging.info("Using testdata in %s", self.get_temp_dir())
    avg_model_dir = self._export_global_average_model()
    image_dir = self._write_random_dataset()

    with flagsaver.flagsaver(
        image_dir=image_dir, tfhub_module=avg_model_dir,
        # This is expeced to fail for this random dataset.
        assert_accuracy_at_least=0.8, **self.DEFAULT_FLAGS):
      with self.assertRaisesRegex(AssertionError, "ACCURACY FAILED"):
        make_image_classifier.main([])

  def testImageSizeForModuleWithFixedInputSize(self):
    model_dir = self._export_global_average_model(has_fixed_input_size=True)
    module_layer = hub.KerasLayer(model_dir)
    self.assertTupleEqual(
        (self.IMAGE_SIZE, self.IMAGE_SIZE),
        make_image_classifier._image_size_for_module(module_layer, None))
    self.assertTupleEqual(
        (self.IMAGE_SIZE, self.IMAGE_SIZE),
        make_image_classifier._image_size_for_module(module_layer,
                                                     self.IMAGE_SIZE))
    with self.assertRaisesRegex(ValueError, "image size"):
      make_image_classifier._image_size_for_module(
          module_layer, self.IMAGE_SIZE + 1)

  def testImageSizeForModuleWithVariableInputSize(self):
    model_dir = self._export_global_average_model(has_fixed_input_size=False)
    module_layer = hub.KerasLayer(model_dir)
    self.assertTupleEqual(
        (self.IMAGE_SIZE, self.IMAGE_SIZE),
        make_image_classifier._image_size_for_module(module_layer,
                                                     self.IMAGE_SIZE))
    self.assertTupleEqual(
        (2 * self.IMAGE_SIZE, 2 * self.IMAGE_SIZE),
        make_image_classifier._image_size_for_module(module_layer,
                                                     2 * self.IMAGE_SIZE))
    with self.assertRaisesRegex(ValueError, "none"):
      make_image_classifier._image_size_for_module(module_layer, None)


if __name__ == "__main__":
  try:
    make_image_classifier._ensure_tf2()
  except ImportError as e:
    print("Skipping tests:", str(e))
    sys.exit(0)
  tf.test.main()
