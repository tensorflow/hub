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
"""Tests for tensorflow_hub.tensor_info."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_hub import tensor_info


def _make_signature(inputs, outputs, name=None):
  input_info = {
      input_name: tf.compat.v1.saved_model.utils.build_tensor_info(tensor)
      for input_name, tensor in inputs.items()
  }
  output_info = {
      output_name: tf.compat.v1.saved_model.utils.build_tensor_info(tensor)
      for output_name, tensor in outputs.items()
  }
  return tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
      input_info, output_info, name)


class TensorInfoTest(tf.test.TestCase):

  def testParsingTensorInfoProtoMaps(self):
    with tf.compat.v1.Graph().as_default():
      sig = _make_signature({
          "x": tf.compat.v1.placeholder(tf.string, [2]),
      }, {
          "y": tf.compat.v1.placeholder(tf.int32, [2]),
          "z": tf.compat.v1.sparse_placeholder(tf.float32, [2, 10]),
      })

      inputs = tensor_info.parse_tensor_info_map(sig.inputs)
      self.assertEqual(set(inputs.keys()), set(["x"]))
      self.assertEqual(inputs["x"].get_shape(), [2])
      self.assertEqual(inputs["x"].dtype, tf.string)
      self.assertFalse(inputs["x"].is_sparse)

      outputs = tensor_info.parse_tensor_info_map(sig.outputs)
      self.assertEqual(set(outputs.keys()), set(["y", "z"]))
      self.assertEqual(outputs["y"].get_shape(), [2])
      self.assertEqual(outputs["y"].dtype, tf.int32)
      self.assertFalse(outputs["y"].is_sparse)

      self.assertEqual(outputs["z"].get_shape(), [2, 10])
      self.assertEqual(outputs["z"].dtype, tf.float32)
      self.assertTrue(outputs["z"].is_sparse)

  def testRepr(self):
    with tf.compat.v1.Graph().as_default():
      sig = _make_signature({
          "x": tf.compat.v1.placeholder(tf.string, [2]),
      }, {
          "y": tf.compat.v1.placeholder(tf.int32, [2]),
          "z": tf.compat.v1.sparse_placeholder(tf.float32, [2, 10]),
      })

      outputs = tensor_info.parse_tensor_info_map(sig.outputs)
      self.assertEqual(
          repr(outputs["y"]),
          "<hub.ParsedTensorInfo shape=(2,) dtype=int32 is_sparse=False>")
      self.assertEqual(
          repr(outputs["z"]),
          "<hub.ParsedTensorInfo shape=(2, 10) dtype=float32 is_sparse=True>")


  def testMatchingTensorInfoProtoMaps(self):
    with tf.compat.v1.Graph().as_default():
      sig1 = _make_signature({
          "x": tf.compat.v1.placeholder(tf.int32, [2]),
      }, {
          "x": tf.compat.v1.placeholder(tf.int32, [2]),
      })

      sig2 = _make_signature({
          "x": tf.compat.v1.placeholder(tf.int32, [2]),
      }, {
          "x": tf.compat.v1.sparse_placeholder(tf.int64, [2]),
      })
      self.assertTrue(
          tensor_info.tensor_info_proto_maps_match(sig1.inputs, sig2.inputs))
      self.assertFalse(
          tensor_info.tensor_info_proto_maps_match(sig1.outputs, sig2.outputs))

      sig3 = _make_signature({
          "x": tf.compat.v1.placeholder(tf.int32, [None]),
      }, {
          "x": tf.compat.v1.placeholder(tf.int32, [2]),
      })
      self.assertFalse(
          tensor_info.tensor_info_proto_maps_match(sig1.inputs, sig3.inputs))
      self.assertTrue(
          tensor_info.tensor_info_proto_maps_match(sig1.outputs, sig3.outputs))

  def testBuildInputMap(self):
    with tf.compat.v1.Graph().as_default():
      x = tf.compat.v1.placeholder(tf.int32, [2])
      y = tf.compat.v1.sparse_placeholder(tf.string, [None])
      sig = _make_signature({"x": x, "y": y}, {})

      input_map = tensor_info.build_input_map(sig.inputs, {"x": x, "y": y})
      self.assertEqual(len(input_map), 4)
      self.assertEqual(input_map[x.name], x)
      self.assertEqual(input_map[y.indices.name], y.indices)
      self.assertEqual(input_map[y.values.name], y.values)
      self.assertEqual(input_map[y.dense_shape.name], y.dense_shape)

  def testBuildOutputMap(self):
    with tf.compat.v1.Graph().as_default():
      x = tf.compat.v1.placeholder(tf.int32, [2])
      y = tf.compat.v1.sparse_placeholder(tf.string, [None])
      sig = _make_signature({}, {"x": x, "y": y})

      def _get_tensor(name):
        return tf.compat.v1.get_default_graph().get_tensor_by_name(name)

      output_map = tensor_info.build_output_map(sig.outputs, _get_tensor)
      self.assertEqual(len(output_map), 2)
      self.assertEqual(output_map["x"], x)
      self.assertEqual(output_map["y"].indices, y.indices)
      self.assertEqual(output_map["y"].values, y.values)
      self.assertEqual(output_map["y"].dense_shape, y.dense_shape)

  def testConvertTensors(self):
    with tf.compat.v1.Graph().as_default():
      a = tf.compat.v1.placeholder(tf.int32, [None])
      protomap = _make_signature({"a": a}, {}).inputs
      targets = tensor_info.parse_tensor_info_map(protomap)

      # convert constant
      in0 = [1, 2, 3]
      output = tensor_info.convert_dict_to_compatible_tensor({"a": in0},
                                                             targets)
      self.assertEqual(output["a"].dtype, a.dtype)

      # check sparsity
      in1 = tf.compat.v1.sparse_placeholder(tf.int32, [])
      with self.assertRaisesRegexp(TypeError, "dense"):
        tensor_info.convert_dict_to_compatible_tensor({"a": in1}, targets)


if __name__ == "__main__":
  tf.test.main()
