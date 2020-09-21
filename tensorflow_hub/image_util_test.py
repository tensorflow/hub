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
"""Tests for image_util.py."""

import tensorflow as tf

from tensorflow_hub import image_util
from tensorflow_hub import module
from tensorflow_hub import native_module


def image_module_fn():
  images = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 2, 4, 3])
  sum_by_channels = tf.reduce_sum(images, [1, 2])
  sum_all = tf.reduce_sum(images, [1, 2, 3])
  native_module.add_signature(inputs=dict(images=images),
                              outputs=dict(default=sum_all,
                                           sum_by_channels=sum_by_channels))


def image_module_fn_with_info():
  images = tf.compat.v1.placeholder(dtype=tf.float32,
                                    shape=[None, None, None, 3])
  sum_all = tf.reduce_sum(images, [1, 2, 3])
  native_module.add_signature(inputs=dict(images=images),
                              outputs=dict(default=sum_all))
  image_module_info = image_util.ImageModuleInfo()
  size = image_module_info.default_image_size
  size.height, size.width = 2, 4
  image_util.attach_image_module_info(image_module_info)


class ImageModuleTest(tf.test.TestCase):

  def testGetExpectedImageSizeFromShape(self):
    with tf.Graph().as_default():
      spec = native_module.create_module_spec(image_module_fn)
      self.assertAllEqual(image_util.get_expected_image_size(spec), [2, 4])
      m = module.Module(spec)
      self.assertAllEqual(image_util.get_expected_image_size(m), [2, 4])

  def testGetExpectedImageSizeFromImageModuleInfo(self):
    with tf.Graph().as_default():
      spec = native_module.create_module_spec(image_module_fn_with_info)
      self.assertAllEqual(image_util.get_expected_image_size(spec), [2, 4])
      m = module.Module(spec)
      self.assertAllEqual(image_util.get_expected_image_size(m), [2, 4])

  def testGetNumImageChannels(self):
    with tf.Graph().as_default():
      spec = native_module.create_module_spec(image_module_fn)
      self.assertEqual(image_util.get_num_image_channels(spec), 3)
      m = module.Module(spec)
      self.assertEqual(image_util.get_num_image_channels(m), 3)


if __name__ == "__main__":
  tf.test.main()
