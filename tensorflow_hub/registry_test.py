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
"""Tests for tensorflow_hub.registry."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_hub import registry


class TestImpl(object):

  def __init__(self, is_supported, execute):
    self._is_supported = is_supported
    self._execute = execute

  def is_supported(self, *args, **kwargs):
    return self._is_supported(*args, **kwargs)

  def __call__(self, *args, **kwargs):
    return self._execute(*args, **kwargs)


class RegistryTest(tf.test.TestCase):

  def testResolveInReverseOrder(self):
    def fail(_):
      raise AssertionError("should not be called")

    r = registry.MultiImplRegister("test")
    r.add_implementation(TestImpl(lambda _: True, lambda _: 0))
    r.add_implementation(TestImpl(lambda x: x == 1, lambda _: 100))
    r.add_implementation(TestImpl(lambda x: x == 2, fail))
    r.add_implementation(TestImpl(lambda x: x == 2, lambda _: 200))

    self.assertEqual(r(0), 0)
    self.assertEqual(r(1), 100)
    self.assertEqual(r(2), 200)


if __name__ == "__main__":
  tf.test.main()
