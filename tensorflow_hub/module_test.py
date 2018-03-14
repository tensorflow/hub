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
"""Unit tests for tensorflow_hub.module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_hub import module
from tensorflow_hub import module_impl
from tensorflow_hub import module_spec
from tensorflow_hub import tensor_info


class TestConvertInputsOutputs(tf.test.TestCase):

  def testSingleInput(self):
    inputs_info = {
        "x": tensor_info.ParsedTensorInfo(
            tf.float32,
            tf.TensorShape([None]),
            is_sparse=False),
    }
    def _check(dict_inputs):
      self.assertEqual(len(dict_inputs), 1)
      self.assertEqual(dict_inputs["x"].dtype, tf.float32)
      self.assertTrue(dict_inputs["x"].shape.is_compatible_with([None]))

    _check(module._prepare_dict_inputs([1, 2], inputs_info))
    _check(module._prepare_dict_inputs({"x": [1, 2]}, inputs_info))

    with self.assertRaisesRegexp(TypeError, r"missing \['x'\]"):
      module._prepare_dict_inputs(None, inputs_info)

    with self.assertRaisesRegexp(TypeError, r"extra given \['y'\]"):
      module._prepare_dict_inputs({"x": [1, 2], "y": [1, 2]}, inputs_info)

  def testNoInputs(self):
    self.assertEqual(module._prepare_dict_inputs(None, {}), {})
    self.assertEqual(module._prepare_dict_inputs({}, {}), {})

    with self.assertRaisesRegexp(TypeError, "expects no inputs"):
      module._prepare_dict_inputs([None], {})

    with self.assertRaisesRegexp(TypeError, "expects no inputs"):
      module._prepare_dict_inputs(1, {})

    with self.assertRaisesRegexp(TypeError, r"extra given \['x'\]"):
      module._prepare_dict_inputs({"x": 1}, {})

  def testMultipleInputs(self):
    inputs_info = {
        "x": tensor_info.ParsedTensorInfo(
            tf.float32,
            tf.TensorShape([None]),
            is_sparse=False),
        "y": tensor_info.ParsedTensorInfo(
            tf.float32,
            tf.TensorShape([None]),
            is_sparse=False),
    }
    def _check(dict_inputs):
      self.assertEqual(len(dict_inputs), 2)
      for key in ("x", "y"):
        self.assertEqual(dict_inputs[key].dtype, tf.float32)
        self.assertTrue(dict_inputs[key].shape.is_compatible_with([None]))

    _check(module._prepare_dict_inputs({"x": [1, 2], "y": [1, 2]},
                                       inputs_info))

    with self.assertRaisesRegexp(TypeError, r"missing \['x', 'y'\]"):
      module._prepare_dict_inputs(None, inputs_info)
    with self.assertRaisesRegexp(TypeError, r"missing \['x', 'y'\]"):
      module._prepare_dict_inputs({}, inputs_info)
    with self.assertRaisesRegexp(TypeError, r"missing \['x', 'y'\]"):
      module._prepare_dict_inputs({"z": 1}, inputs_info)

    with self.assertRaisesRegexp(
        TypeError, "Signature expects multiple inputs. Use a dict."):
      module._prepare_dict_inputs(1, inputs_info)

  def testOutputWithDefault(self):
    outputs = {"default": "result", "extra": "dbg info"}
    self.assertEquals(module._prepare_outputs(outputs, as_dict=False), "result")
    self.assertEquals(module._prepare_outputs(outputs, as_dict=True), outputs)

  def testDictOutput(self):
    outputs = {"x": 1, "y": 2}
    self.assertEquals(module._prepare_outputs(outputs, as_dict=True), outputs)
    with self.assertRaisesRegexp(TypeError, r"Use as_dict=True."):
      self.assertEquals(module._prepare_outputs(outputs, as_dict=False),
                        outputs)


class GetStateScopeTest(tf.test.TestCase):

  def testGetStateScope(self):
    self.assertEqual(module._try_get_state_scope("a"), "a/")
    self.assertEqual(module._try_get_state_scope("a"), "a_1/")

  def testGetStateScope_UsesVariableScope(self):
    self.assertEqual(module._try_get_state_scope("a"), "a/")
    with tf.variable_scope(None, default_name="a") as vs:
      self.assertEqual(vs.name, "a_1")

  def testGetStateScope_UsesNameScope(self):
    self.assertEqual(module._try_get_state_scope("a"), "a/")
    with tf.name_scope("a") as ns:
      self.assertEqual(ns, "a_1/")

  def testGetStateScope_UnusedNameScope(self):
    self.assertEqual(module._try_get_state_scope("a", False), "a/")
    with tf.name_scope("a") as ns:
      self.assertEqual(ns, "a/")

    self.assertEqual(module._try_get_state_scope("a", False), "a_1/")
    with tf.name_scope("a") as ns:
      self.assertEqual(ns, "a_1/")

  def testGetStateScope_AlreadyUsedNameScope(self):
    with tf.name_scope("a"):
      pass
    with self.assertRaisesRegexp(RuntimeError, "name_scope was already taken"):
      module._try_get_state_scope("a", False)

  def testGetStateScopeWithActiveScopes(self):
    with tf.Graph().as_default():
      with tf.name_scope("foo"):
        abs_scope = module._try_get_state_scope("a", False)
        self.assertEqual(abs_scope, "a/")
        with tf.name_scope(abs_scope) as ns:
          self.assertEqual(ns, "a/")

    with tf.Graph().as_default():
      with tf.variable_scope("vs"):
        self.assertEqual(module._try_get_state_scope("a", False), "vs/a/")
        with tf.name_scope(name="a") as ns:
          self.assertEqual(ns, "vs/a/")

    with tf.Graph().as_default():
      with tf.name_scope("foo"):
        with tf.variable_scope("vs"):
          self.assertEquals(module._try_get_state_scope("a", False), "vs/a/")


class _ModuleSpec(module_spec.ModuleSpec):

  def get_tags(self):
    return [set(), set(["special"])]

  def get_signature_names(self, tags=None):
    if tags == set(["special"]):
      return iter(["default", "extra"])
    else:
      return iter(["default"])

  def get_input_info_dict(self, signature=None, tags=None):
    result = {
        "x": tensor_info.ParsedTensorInfo(
            tf.float32,
            tf.TensorShape([None]),
            is_sparse=False),
    }
    if tags == set(["special"]) and signature == "extra":
      result["y"] = result["x"]
    return result

  def get_output_info_dict(self, signature=None, tags=None):
    result = {
        "default": tensor_info.ParsedTensorInfo(
            tf.float32,
            tf.TensorShape([None]),
            is_sparse=False),
    }
    if tags == set(["special"]) and signature == "extra":
      result["z"] = result["default"]
    return result

  def _create_impl(self, name, trainable, tags):
    return _ModuleImpl(name, trainable)


class _ModuleImpl(module_impl.ModuleImpl):

  def __init__(self, name, trainable):
    super(_ModuleImpl, self).__init__()
    with tf.variable_scope(name):
      pass

  def create_apply_graph(self, signature, inputs, name):
    with tf.name_scope(name):
      result = {"default": 2 * inputs["x"]}
      if signature == "extra":
        result["z"] = 2 * inputs["x"] + 3 * inputs["y"]
      return result

  def export(self, path, session):
    raise NotImplementedError()

  @property
  def variable_map(self):
    raise NotImplementedError()


class ModuleTest(tf.test.TestCase):

  def testModuleSingleInput(self):
    m = module.Module(_ModuleSpec())
    result = m([1, 2])
    with tf.Session() as session:
      self.assertAllEqual(session.run(result), [2, 4])

  def testModuleDictInput(self):
    m = module.Module(_ModuleSpec())
    result = m({"x": [1, 2]})
    with tf.Session() as session:
      self.assertAllEqual(session.run(result), [2, 4])

  def testModuleDictOutput(self):
    m = module.Module(_ModuleSpec())
    result = m([1, 2], as_dict=True)
    self.assertTrue(isinstance(result, dict))
    self.assertAllEqual(list(result.keys()), ["default"])

  def testModuleInNestedScope(self):
    with tf.variable_scope("foo"):
      m = module.Module(_ModuleSpec())
      result = m([1, 2])
    with tf.Session() as session:
      self.assertAllEqual(session.run(result), [2, 4])

  def testModuleInterfaceGettersDefaultSignatureAndTags(self):
    m = module.Module(_ModuleSpec())
    self.assertItemsEqual(m.get_signature_names(), ["default"])
    self.assertItemsEqual(m.get_input_info_dict().keys(), ["x"])
    self.assertItemsEqual(m.get_output_info_dict().keys(), ["default"])

  def testModuleInterfaceGettersExplicitSignatureAndTags(self):
    """Tests that tags from Module(...) apply to module.get_*()."""
    m = module.Module(_ModuleSpec(), tags={"special"})
    self.assertItemsEqual(m.get_signature_names(), ["default", "extra"])
    self.assertItemsEqual(m.get_input_info_dict(signature="extra").keys(),
                          ["x", "y"])
    self.assertItemsEqual(m.get_output_info_dict(signature="extra").keys(),
                          ["z", "default"])

if __name__ == "__main__":
  tf.test.main()
