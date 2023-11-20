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
"""Tests for tensorflow_hub.module_v2."""

import os
from absl.testing import parameterized
import tensorflow as tf
from tensorflow_hub import module_v2


def _save_plus_one_saved_model_v2(path):
  obj = tf.train.Checkpoint()

  @tf.function(input_signature=[tf.TensorSpec(None, dtype=tf.float32)])
  def plus_one(x):
    return x + 1

  obj.__call__ = plus_one
  tf.saved_model.save(obj, path)


class ModuleV2Test(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('v2_implicit_tags', None, False),
      ('v2_explicit_tags', ['serve'], False),
  )
  def test_load(self, tags, is_hub_module_v1):
    module_name = 'saved_model_v2_mini'
    export_dir = os.path.join(self.get_temp_dir(), module_name)
    _save_plus_one_saved_model_v2(export_dir)
    m = module_v2.load(export_dir, tags)
    self.assertEqual(m._is_hub_module_v1, is_hub_module_v1)

  def test_load_incomplete_model_fails(self):
    temp_dir = self.create_tempdir().full_path
    self.create_tempfile(os.path.join(temp_dir, 'variables.txt'))

    with self.assertRaisesRegex(ValueError, 'contains neither'):
      module_v2.load(temp_dir)

  def test_load_without_string(self):
    with self.assertRaisesRegex(ValueError, 'Expected a string, got.*'):
      module_v2.load(0)


if __name__ == '__main__':
  # In TF 1.15.x, we need to enable V2-like behavior, notably eager execution.
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
