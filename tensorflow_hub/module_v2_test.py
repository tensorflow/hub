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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint:disable=g-import-not-at-top,g-statement-before-imports
try:
  import mock as mock
except ImportError:
  import unittest.mock as mock
# pylint:disable=g-import-not-at-top,g-statement-before-imports

import os
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_hub import module_v2


def _save_plus_one_saved_model_v2(path):
  obj = tf.train.Checkpoint()

  @tf.function(input_signature=[tf.TensorSpec(None, dtype=tf.float32)])
  def plus_one(x):
    return x + 1

  obj.__call__ = plus_one
  tf.saved_model.save(obj, path)


def _save_plus_one_hub_module_v1(path):

  def plus_one():
    x = tf.compat.v1.placeholder(dtype=tf.float32, name='x')
    y = x + 1
    hub.add_signature(inputs=x, outputs=y)

  spec = hub.create_module_spec(plus_one)

  with tf.compat.v1.Graph().as_default():
    module = hub.Module(spec, trainable=True)
    with tf.compat.v1.Session() as session:
      session.run(tf.compat.v1.global_variables_initializer())
      module.export(path, session)


class ModuleV2Test(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('v1_implicit_tags', 'hub_module_v1_mini', None, True),
      ('v1_explicit_tags', 'hub_module_v1_mini', [], True),
      ('v2_implicit_tags', 'saved_model_v2_mini', None, False),
      ('v2_explicit_tags', 'saved_model_v2_mini', ['serve'], False),
  )
  def test_load(self, module_name, tags, is_hub_module_v1):
    export_dir = os.path.join(self.get_temp_dir(), module_name)
    if module_name == 'hub_module_v1_mini':
      _save_plus_one_hub_module_v1(export_dir)
    else:
      _save_plus_one_saved_model_v2(export_dir)
    m = module_v2.load(export_dir, tags)
    self.assertEqual(m._is_hub_module_v1, is_hub_module_v1)

  @mock.patch.object(module_v2, 'tf_v1')
  def test_load_with_old_tensorflow_raises_error(self, tf_v1_mock):
    tf_v1_mock.saved_model = None
    with self.assertRaises(NotImplementedError):
      module_v2.load('dummy_module_name')

  def test_load_without_string(self):
    with self.assertRaisesRegex(ValueError, 'Expected a string, got.*'):
      module_v2.load(0)


if __name__ == '__main__':
  # In TF 1.15.x, we need to enable V2-like behavior, notably eager execution.
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
