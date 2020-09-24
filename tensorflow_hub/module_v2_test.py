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


def _save_sparse_plus_one_hub_module_v1(path):

  def plus_one():
    x = tf.compat.v1.sparse.placeholder(dtype=tf.float32, name='x')
    y = tf.identity(tf.SparseTensor(x.indices, x.values + 1, x.dense_shape))
    hub.add_signature(inputs=x, outputs=y)

  spec = hub.create_module_spec(plus_one)

  with tf.compat.v1.Graph().as_default():
    module = hub.Module(spec, trainable=True)
    with tf.compat.v1.Session() as session:
      session.run(tf.compat.v1.global_variables_initializer())
      module.export(path, session)


def _save_ragged_plus_one_hub_module_v1(path):

  def plus_one():
    x = tf.compat.v1.ragged.placeholder(
        dtype=tf.float32, ragged_rank=1, value_shape=[], name='x')
    y = tf.identity(x + 1)
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

  def test_load_sparse(self):
    if any(tf.__version__.startswith(bad) for bad in ['1.', '2.0.']):
      self.skipTest('load_v1_in_v2 did not handle sparse tensors correctly'
                    'in TensorFlow version %r.' % (tf.__version__,))
    export_dir = os.path.join(self.get_temp_dir(), 'sparse')
    _save_sparse_plus_one_hub_module_v1(export_dir)
    m = module_v2.load(export_dir)
    self.assertTrue(m._is_hub_module_v1)
    plus_one = m.signatures['default']
    st = tf.sparse.from_dense([[1.0, 2.0, 0.0], [0.0, 3.0, 0.0]])
    actual = plus_one(
        default_indices=st.indices,
        default_values=st.values,
        default_dense_shape=st.dense_shape)['default']
    expected = [2.0, 3.0, 4.0]
    self.assertAllEqual(actual.values, expected)

  def test_load_ragged(self):
    if any(tf.__version__.startswith(bad) for bad in
           ['1.', '2.0.', '2.1.', '2.2.', '2.3.']):
      self.skipTest('load_v1_in_v2 did not handle composite tensors correctly'
                    'in TensorFlow version %r.' % (tf.__version__,))
    export_dir = os.path.join(self.get_temp_dir(), 'ragged')
    _save_ragged_plus_one_hub_module_v1(export_dir)
    m = module_v2.load(export_dir)
    self.assertTrue(m._is_hub_module_v1)
    plus_one = m.signatures['default']
    rt = tf.ragged.constant([[1.0, 8.0], [3.0]])
    actual = plus_one(default_component_0=rt.values,
                      default_component_1=rt.row_splits)['default']
    expected = [2.0, 9.0, 4.0]
    self.assertAllEqual(actual.values, expected)

  def test_load_without_string(self):
    with self.assertRaisesRegex(ValueError, 'Expected a string, got.*'):
      module_v2.load(0)


if __name__ == '__main__':
  # In TF 1.15.x, we need to enable V2-like behavior, notably eager execution.
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
