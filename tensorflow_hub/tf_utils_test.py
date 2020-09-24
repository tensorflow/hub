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
"""Tests for tensorflow_hub.tf_utils."""

import tensorflow as tf
from tensorflow_hub import tf_utils


class TfUtilsTest(tf.test.TestCase):

  def testIsCompositeTensor(self):
    ragged_tensor = tf.ragged.constant([[1, 2], [3]])
    self.assertTrue(tf_utils.is_composite_tensor(ragged_tensor))

    sparse_tensor = tf.SparseTensor([[0, 2], [3, 2]], [5, 6], [10, 10])
    self.assertTrue(tf_utils.is_composite_tensor(sparse_tensor))

    tensor = tf.constant([1, 2, 3])
    self.assertFalse(tf_utils.is_composite_tensor(tensor))

  def testGetCompositeTensorTypeSpec(self):
    ragged_tensor = tf.ragged.constant([[1, 2], [3]])
    self.assertIsInstance(
        tf_utils.get_composite_tensor_type_spec(ragged_tensor),
        tf.RaggedTensorSpec)

    sparse_tensor = tf.SparseTensor([[0, 2], [3, 2]], [5, 6], [10, 10])
    self.assertIsInstance(
        tf_utils.get_composite_tensor_type_spec(sparse_tensor),
        tf.SparseTensorSpec)

    tensor = tf.constant([1, 2, 3])
    self.assertIs(tf_utils.get_composite_tensor_type_spec(tensor), None)


if __name__ == "__main__":
  tf.test.main()
