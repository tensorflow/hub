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
      input_name: tf.saved_model.utils.build_tensor_info(tensor)
      for input_name, tensor in inputs.items()
  }
  output_info = {
      output_name: tf.saved_model.utils.build_tensor_info(tensor)
      for output_name, tensor in outputs.items()
  }
  return tf.saved_model.signature_def_utils.build_signature_def(
      input_info, output_info, name)


class TensorInfoTest(tf.test.TestCase):

  def testParsingTensorInfoProtoMaps(self):
    sig = _make_signature({
        "x": tf.placeholder(tf.string, [2]),
    }, {
        "y": tf.placeholder(tf.int32, [2]),
        "z": tf.sparse_placeholder(tf.float32, [2, 10]),
    })

    inputs = tensor_info.parse_tensor_info_map(sig.inputs)
    self.assertEquals(set(inputs.keys()), set(["x"]))
    self.assertEquals(inputs["x"].get_shape(), [2])
    self.assertEquals(inputs["x"].dtype, tf.string)
    self.assertFalse(inputs["x"].is_sparse)

    outputs = tensor_info.parse_tensor_info_map(sig.outputs)
    self.assertEquals(set(outputs.keys()), set(["y", "z"]))
    self.assertEquals(outputs["y"].get_shape(), [2])
    self.assertEquals(outputs["y"].dtype, tf.int32)
    self.assertFalse(outputs["y"].is_sparse)

    self.assertEquals(outputs["z"].get_shape(), [2, 10])
    self.assertEquals(outputs["z"].dtype, tf.float32)
    self.assertTrue(outputs["z"].is_sparse)

  def testRepr(self):
    sig = _make_signature({
        "x": tf.placeholder(tf.string, [2]),
    }, {
        "y": tf.placeholder(tf.int32, [2]),
        "z": tf.sparse_placeholder(tf.float32, [2, 10]),
    })

    outputs = tensor_info.parse_tensor_info_map(sig.outputs)
    self.assertEquals(
        repr(outputs["y"]),
        "<hub.ParsedTensorInfo shape=(2,) dtype=int32 is_sparse=False>")
    self.assertEquals(
        repr(outputs["z"]),
        "<hub.ParsedTensorInfo shape=(2, 10) dtype=float32 is_sparse=True>")


  def testMatchingTensorInfoProtoMaps(self):
    sig1 = _make_signature({
        "x": tf.placeholder(tf.int32, [2]),
    }, {
        "x": tf.placeholder(tf.int32, [2]),
    })

    sig2 = _make_signature({
        "x": tf.placeholder(tf.int32, [2]),
    }, {
        "x": tf.sparse_placeholder(tf.int64, [2]),
    })
    self.assertTrue(
        tensor_info.tensor_info_proto_maps_match(sig1.inputs, sig2.inputs))
    self.assertFalse(
        tensor_info.tensor_info_proto_maps_match(sig1.outputs, sig2.outputs))

    sig3 = _make_signature({
        "x": tf.placeholder(tf.int32, [None]),
    }, {
        "x": tf.placeholder(tf.int32, [2]),
    })
    self.assertFalse(
        tensor_info.tensor_info_proto_maps_match(sig1.inputs, sig3.inputs))
    self.assertTrue(
        tensor_info.tensor_info_proto_maps_match(sig1.outputs, sig3.outputs))

  def testBuildInputMap(self):
    x = tf.placeholder(tf.int32, [2])
    y = tf.sparse_placeholder(tf.string, [None])
    sig = _make_signature({"x": x, "y": y}, {})

    input_map = tensor_info.build_input_map(sig.inputs, {"x": x, "y": y})
    self.assertEquals(len(input_map), 4)
    self.assertEquals(input_map[x.name], x)
    self.assertEquals(input_map[y.indices.name], y.indices)
    self.assertEquals(input_map[y.values.name], y.values)
    self.assertEquals(input_map[y.dense_shape.name], y.dense_shape)

  def testBuildOutputMap(self):
    x = tf.placeholder(tf.int32, [2])
    y = tf.sparse_placeholder(tf.string, [None])
    sig = _make_signature({}, {"x": x, "y": y})

    def _get_tensor(name):
      return tf.get_default_graph().get_tensor_by_name(name)

    output_map = tensor_info.build_output_map(sig.outputs, _get_tensor)
    self.assertEquals(len(output_map), 2)
    self.assertEquals(output_map["x"], x)
    self.assertEquals(output_map["y"].indices, y.indices)
    self.assertEquals(output_map["y"].values, y.values)
    self.assertEquals(output_map["y"].dense_shape, y.dense_shape)

  def testConvertTensors(self):
    a = tf.placeholder(tf.int32, [None])
    protomap = _make_signature({"a": a}, {}).inputs

    # convert constant
    in0 = [1, 2, 3]
    output = tensor_info.convert_to_input_tensors(protomap, {"a": in0})
    self.assertEquals(output["a"].dtype, a.dtype)

    # check sparsity
    in1 = tf.sparse_placeholder(tf.int32, [])
    with self.assertRaisesRegexp(TypeError, "dense"):
      tensor_info.convert_to_input_tensors(protomap, {"a": in1})

    # check args mismatch
    with self.assertRaisesRegexp(TypeError, "missing"):
      tensor_info.convert_to_input_tensors(protomap, {"b": in1})


if __name__ == "__main__":
  tf.test.main()
