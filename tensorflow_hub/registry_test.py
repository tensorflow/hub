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

import six
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


def fail_fn(_):
  raise AssertionError("fail_fn should not be called")


class RegistryTest(tf.test.TestCase):

  def testResolveAlwaysSupported(self):
    r = registry.MultiImplRegister("test")
    r.add_implementation(TestImpl(lambda _: True, lambda _: 100))
    r.add_implementation(TestImpl(lambda _: False, fail_fn))

    self.assertEqual(r(1), 100)

  def testResolveWhenSupported(self):
    r = registry.MultiImplRegister("test")
    r.add_implementation(TestImpl(lambda x: x == 1, lambda _: 100))
    r.add_implementation(TestImpl(lambda x: x == 2, lambda _: 200))
    r.add_implementation(TestImpl(lambda _: False, fail_fn))

    self.assertEqual(r(1), 100)
    self.assertEqual(r(2), 200)

  def testLogWhenContainsNotSupported(self):
    if six.PY2:
      return
    with self.assertLogs(level="INFO") as logs:
      r = registry.MultiImplRegister("test")
      r.add_implementation(TestImpl(lambda x: x == 1, lambda _: 100))
      r.add_implementation(TestImpl(lambda x: x == 2, lambda _: 200))
      r.add_implementation(TestImpl(lambda _: False, fail_fn))

      r(2)

    self.assertEqual(
        logs.output,
        ["INFO:absl:test TestImpl does not support the provided handle."])

  def testResolveInReverseOrder(self):
    r = registry.MultiImplRegister("test")
    r.add_implementation(TestImpl(lambda _: True, fail_fn))
    r.add_implementation(TestImpl(lambda _: True, lambda _: 100))

    self.assertEqual(r(1), 100)

  def testResolveThrowsNoSupportedImplementation(self):
    r = registry.MultiImplRegister("test")
    r.add_implementation(TestImpl(lambda _: False, lambda _: 100))

    self.assertRaisesRegex(
        RuntimeError,
        "Missing implementation that supports: test\(\*\(1,\), \*\*{}\)", r, 1)


if __name__ == "__main__":
  tf.test.main()
