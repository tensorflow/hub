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

import os

import tensorflow as tf
from tensorflow_hub import saved_model_lib


def _instantiate_meta_graph(saved_model_handler, tags=None):
  """Loads a MetaGraph from a SavedModelHandler into a new Graph."""
  meta_graph = saved_model_handler.get_meta_graph(tags)
  with tf.Graph().as_default() as graph:
    tf.train.import_meta_graph(meta_graph, import_scope="")
  return graph


def _write_string_to_file(path, contents):
  with tf.gfile.Open(path, "w") as f:
    f.write(contents)


def _read_file_to_string(path):
  with tf.gfile.Open(path, "r") as f:
    return f.read()


class SavedModelLibTest(tf.test.TestCase):

  def testAssets(self):
    original_asset_file = os.path.join(self.get_temp_dir(), "hello.txt")
    _write_string_to_file(original_asset_file, "hello world")

    with tf.Graph().as_default() as graph:
      asset_tensor = tf.constant(original_asset_file, name="file")
      graph.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, asset_tensor)
      saved_model_lib.add_signature("default", {}, {"default": asset_tensor})

    handler = saved_model_lib.SavedModelHandler()
    handler.add_graph_copy(graph)

    export_dir = os.path.join(self.get_temp_dir(), "exported")
    handler.export(export_dir)

    # Check that asset file got written to the expected place:
    exported_asset_file = os.path.join(export_dir, "assets", "hello.txt")
    self.assertTrue(tf.gfile.Exists(exported_asset_file))

    loaded_handler = saved_model_lib.load(export_dir)
    with _instantiate_meta_graph(loaded_handler).as_default():
      with tf.Session() as sess:
        self.assertEqual(sess.run("file:0"),
                         tf.compat.as_bytes(exported_asset_file))

  def testWithMultipleAssetsWithSameBasename(self):
    tmp_asset_dir = os.path.join(self.get_temp_dir(), "asset")
    file_a = os.path.join(tmp_asset_dir, "a", "hello.txt")
    file_b = os.path.join(tmp_asset_dir, "b", "hello.txt")
    tf.gfile.MakeDirs(os.path.dirname(file_a))
    tf.gfile.MakeDirs(os.path.dirname(file_b))
    _write_string_to_file(file_a, "hello A")
    _write_string_to_file(file_b, "hello B")
    with tf.Graph().as_default() as graph:
      asset_a = tf.constant(file_a, name="file_a")
      asset_b = tf.constant(file_b, name="file_b")
      graph.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, asset_a)
      graph.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, asset_b)
      saved_model_lib.add_signature("default", {}, {"default": asset_a})

    export_dir = os.path.join(self.get_temp_dir(), "exported")
    handler = saved_model_lib.SavedModelHandler()
    handler.add_graph_copy(graph)
    handler.export(export_dir)
    tf.gfile.DeleteRecursively(tmp_asset_dir)

    loaded_handler = saved_model_lib.load(export_dir)
    with _instantiate_meta_graph(loaded_handler).as_default():
      with tf.Session() as sess:
        self.assertEqual(_read_file_to_string(sess.run("file_a:0")), "hello A")
        self.assertEqual(_read_file_to_string(sess.run("file_b:0")), "hello B")

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
    self.assertTrue(handler.get_meta_graph_copy(["tag1"]) is not None)
    self.assertTrue(handler.get_meta_graph_copy(["tag2", "tag1"]) is not None)
    with self.assertRaises(KeyError):
      handler.get_meta_graph_copy(["tag2"])

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


if __name__ == "__main__":
  tf.test.main()
