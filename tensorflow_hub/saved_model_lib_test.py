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
"""Tests for tensorflow_hub.saved_model_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
from absl import logging
import six

import tensorflow as tf
from tensorflow_hub import saved_model_lib
from tensorflow_hub import tf_v1

from tensorflow.core.protobuf import meta_graph_pb2


def _instantiate_meta_graph(saved_model_handler, tags=None):
  """Loads a MetaGraph from a SavedModelHandler into a new Graph."""
  meta_graph = saved_model_handler.get_meta_graph(tags)
  with tf.Graph().as_default() as graph:
    tf_v1.train.import_meta_graph(meta_graph, import_scope="")
  return graph


def _write_string_to_file(path, contents):
  with tf_v1.gfile.Open(path, "w") as f:
    f.write(contents)


def _read_file_to_string(path):
  with tf_v1.gfile.Open(path, "r") as f:
    return f.read()


class SavedModelLibTest(tf.test.TestCase):

  def setUp(self):
    super(SavedModelLibTest, self).setUp()
    logging.set_verbosity(logging.DEBUG)

  def testAssets(self):
    original_asset_file = os.path.join(self.get_temp_dir(), "hello.txt")
    _write_string_to_file(original_asset_file, "hello world")

    with tf.Graph().as_default() as graph:
      asset_tensor = tf.constant(original_asset_file, name="file")
      graph.add_to_collection(tf_v1.GraphKeys.ASSET_FILEPATHS, asset_tensor)
      saved_model_lib.add_signature("default", {}, {"default": asset_tensor})

    handler = saved_model_lib.SavedModelHandler()
    handler.add_graph_copy(graph)

    export_dir = os.path.join(self.get_temp_dir(), "exported")
    handler.export(export_dir)

    # Check that asset file got written to the expected place:
    exported_asset_file = os.path.join(export_dir, "assets", "hello.txt")
    self.assertTrue(tf_v1.gfile.Exists(exported_asset_file))

    loaded_handler = saved_model_lib.load(export_dir)
    with _instantiate_meta_graph(loaded_handler).as_default():
      with tf_v1.Session() as sess:
        self.assertEqual(sess.run("file:0"),
                         tf.compat.as_bytes(exported_asset_file))

  def testWithMultipleAssetsWithSameBasename(self):
    tmp_asset_dir = os.path.join(self.get_temp_dir(), "asset")
    file_a = os.path.join(tmp_asset_dir, "a", "hello.txt")
    file_b = os.path.join(tmp_asset_dir, "b", "hello.txt")
    tf_v1.gfile.MakeDirs(os.path.dirname(file_a))
    tf_v1.gfile.MakeDirs(os.path.dirname(file_b))
    _write_string_to_file(file_a, "hello A")
    _write_string_to_file(file_b, "hello B")
    with tf.Graph().as_default() as graph:
      asset_a = tf.constant(file_a, name="file_a")
      asset_b = tf.constant(file_b, name="file_b")
      graph.add_to_collection(tf_v1.GraphKeys.ASSET_FILEPATHS, asset_a)
      graph.add_to_collection(tf_v1.GraphKeys.ASSET_FILEPATHS, asset_b)
      saved_model_lib.add_signature("default", {}, {"default": asset_a})

    export_dir = os.path.join(self.get_temp_dir(), "exported")
    handler = saved_model_lib.SavedModelHandler()
    handler.add_graph_copy(graph)
    handler.export(export_dir)
    tf_v1.gfile.DeleteRecursively(tmp_asset_dir)

    loaded_handler = saved_model_lib.load(export_dir)
    with _instantiate_meta_graph(loaded_handler).as_default():
      with tf_v1.Session() as sess:
        self.assertEqual(_read_file_to_string(sess.run("file_a:0")), "hello A")
        self.assertEqual(_read_file_to_string(sess.run("file_b:0")), "hello B")

  def testCreationOfAssetsKeyCollectionIsDeterministic(self):
    tmp_asset_dir = os.path.join(self.get_temp_dir(), "assets")
    tf_v1.gfile.MakeDirs(tmp_asset_dir)
    filenames = [
        os.path.join(tmp_asset_dir, "file%d.txt" % n) for n in range(10)
    ]
    for filename in filenames:
      _write_string_to_file(filename, "I am file %s" % filename)

    with tf.Graph().as_default() as graph:
      assets = [tf.constant(f, name=os.path.basename(f)) for f in filenames]
      for asset in assets:
        graph.add_to_collection(tf_v1.GraphKeys.ASSET_FILEPATHS, asset)
      saved_model_lib.add_signature("default", {}, {"default": assets[0]})

    handler = saved_model_lib.SavedModelHandler()
    handler.add_graph_copy(graph)
    saved_model_proto = copy.deepcopy(handler._proto)
    export_dir = os.path.join(self.get_temp_dir(), "assets_key_test")
    saved_model_lib._make_assets_key_collection(saved_model_proto, export_dir)

    meta_graph = list(saved_model_proto.meta_graphs)[0]
    asset_tensor_names = []
    for asset_any_proto in meta_graph.collection_def[
        tf_v1.saved_model.constants.ASSETS_KEY].any_list.value:
      asset_proto = meta_graph_pb2.AssetFileDef()
      asset_any_proto.Unpack(asset_proto)
      asset_tensor_names.append(asset_proto.tensor_info.name)
    self.assertCountEqual(asset_tensor_names, asset_tensor_names)

  def testSignatures(self):
    with tf.Graph().as_default() as graph:
      input_a = tf.constant(2)
      input_b = tf.constant(3)
      mul = input_a * input_b
      saved_model_lib.add_signature("six", {}, {"out": mul})
      saved_model_lib.add_signature("mul2", {"in": input_b}, {"out": mul})

    handler = saved_model_lib.SavedModelHandler()
    handler.add_graph_copy(graph)

    signatures = handler.get_meta_graph_copy().signature_def
    self.assertEqual(set(signatures.keys()), set(["six", "mul2"]))
    self.assertAllEqual(list(signatures["six"].inputs.keys()), [])
    self.assertAllEqual(list(signatures["six"].outputs.keys()), ["out"])
    self.assertAllEqual(list(signatures["mul2"].inputs.keys()), ["in"])
    self.assertAllEqual(list(signatures["mul2"].outputs.keys()), ["out"])

  def testSignatureImplementationIsInvisible(self):
    with tf.Graph().as_default() as graph:
      saved_model_lib.add_signature("test", {}, {})
      self.assertEqual(graph.get_all_collection_keys(), [])

    handler = saved_model_lib.SavedModelHandler()
    handler.add_graph_copy(graph)
    meta_graph, = handler.meta_graphs
    self.assertEqual(len(meta_graph.collection_def), 0)
    self.assertEqual(len(meta_graph.signature_def), 1)

  def testTags(self):
    with tf.Graph().as_default() as graph:
      saved_model_lib.add_signature("default", {}, {"default": tf.constant(1)})
    handler = saved_model_lib.SavedModelHandler()
    handler.add_graph_copy(graph, ["tag1"])
    handler.add_graph_copy(graph, ["tag1", "tag2"])
    self.assertAllEqual(sorted(handler.get_tags()),
                        sorted([set(["tag1"]), set(["tag1", "tag2"])]))
    self.assertIsNotNone(handler.get_meta_graph_copy(["tag1"]))
    self.assertIsNotNone(handler.get_meta_graph_copy(["tag2", "tag1"]))
    with self.assertRaises(KeyError):
      handler.get_meta_graph_copy(["tag2"])

  def testModuleAttachments(self):
    meta_graph = meta_graph_pb2.MetaGraphDef()
    with tf.Graph().as_default():
      saved_model_lib.attach_bytes("key1", tf.compat.as_bytes("oops"))
      saved_model_lib.attach_bytes("key2", tf.compat.as_bytes("value2"))
      saved_model_lib.attach_bytes("key1", tf.compat.as_bytes("value1"))
      saved_model_lib._export_module_attachments(meta_graph)
    actual = saved_model_lib.get_attached_bytes_map(meta_graph)
    expected = {"key1": tf.compat.as_bytes("value1"),
                "key2": tf.compat.as_bytes("value2")}
    self.assertDictEqual(expected, actual)

  def testNoModuleAttachments(self):
    meta_graph = meta_graph_pb2.MetaGraphDef()
    with tf.Graph().as_default():
      # No calls to attach_bytes.
      saved_model_lib._export_module_attachments(meta_graph)
    actual = saved_model_lib.get_attached_bytes_map(meta_graph)
    self.assertDictEqual({}, actual)
    # Check there were no unwarranted subscript operations.
    self.assertNotIn(saved_model_lib.ATTACHMENT_COLLECTION_SAVED,
                     meta_graph.collection_def)

  def testEmptyCollectionsDoNotShowUpInMetaGraphDef(self):
    with tf.Graph().as_default() as graph:
      tf.Variable("name")
      self.assertEqual(len(graph.get_all_collection_keys()), 2)
      for collection_key in graph.get_all_collection_keys():
        del graph.get_collection_ref(collection_key)[:]
      saved_model_lib.add_signature("default", {}, {"default": tf.constant(1)})

    handler = saved_model_lib.SavedModelHandler()
    handler.add_graph_copy(graph)
    meta_graph, = handler.meta_graphs
    self.assertEqual(len(meta_graph.collection_def), 0)

  def testBadAssets(self):
    if six.PY2:
      return   # PY3 only test. Remove once PY2 is no longer supported.
    original_asset_file = os.path.join(self.get_temp_dir(), str(b"hello.txt"))
    _write_string_to_file(original_asset_file, "hello world")

    with tf.Graph().as_default() as graph:
      asset_tensor = tf.constant(original_asset_file, name="file")
      graph.add_to_collection(tf_v1.GraphKeys.ASSET_FILEPATHS, asset_tensor)
      saved_model_lib.add_signature("default", {}, {"default": asset_tensor})

    handler = saved_model_lib.SavedModelHandler()
    handler.add_graph_copy(graph)

    export_dir = os.path.join(self.get_temp_dir(), "exported")
    handler.export(export_dir)

    self.assertIn("b\'hello.txt\'",
                  tf_v1.gfile.ListDirectory(export_dir + "/assets"))
    # Check that asset file got written to the expected place:
    exported_asset_file = os.path.join(export_dir, "assets", str(b"hello.txt"))
    self.assertTrue(tf_v1.gfile.Exists(exported_asset_file))

    loaded_handler = saved_model_lib.load(export_dir)
    with _instantiate_meta_graph(loaded_handler).as_default():
      with tf_v1.Session() as sess:
        self.assertEqual(sess.run("file:0"),
                         tf.compat.as_bytes(exported_asset_file))


if __name__ == "__main__":
  tf.test.main()
