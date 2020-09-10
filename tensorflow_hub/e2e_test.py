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
"""End-to-end tests for tensorflow_hub."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import tarfile
import tempfile

from absl import logging
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow_hub import test_utils
from tensorflow_hub import tf_utils

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops.lookup_ops import index_to_string_table_from_file
# pylint: enable=g-direct-tensorflow-import


class End2EndTest(tf.test.TestCase):

  def setUp(self):
    super(End2EndTest, self).setUp()
    # Set current directory to test temp directory where we can create
    # files and serve them through the HTTP server.
    os.chdir(self.get_temp_dir())

    self.server_port = test_utils.start_http_server()

  def _stateless_module_fn(self):
    """Simple module that squares an input."""
    x = tf.compat.v1.placeholder(tf.int64)
    y = x*x
    hub.add_signature(inputs=x, outputs=y)

  def _create_tgz(self, export_path, archive_name="test_module.tgz"):
    os.chdir(export_path)

    tar = tarfile.open(archive_name, "w")
    for directory, subdirs, files in tf.compat.v1.gfile.Walk(export_path):
      for subdir in subdirs:
        tar.add(subdir)
      for file_name in files:
        full_path = os.path.join(directory, file_name)
        tar.add(full_path[len(export_path)+1:])
    tar.close()

  def _generate_module(self):
    spec = hub.create_module_spec(self._stateless_module_fn)
    m = hub.Module(spec, name="test_module")
    out = m(10)

    export_path = os.path.join(self.get_temp_dir(), "module")
    with tf.compat.v1.Session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      self.assertAllClose(sess.run(out), 100)
      m.export(export_path, sess)

    self._create_tgz(export_path)

  def test_http_locations(self):
    with tf.Graph().as_default():
      self._generate_module()

      m = hub.Module("http://localhost:%d/test_module.tgz" % self.server_port)
      out = m(11)
      with tf.compat.v1.Session() as sess:
        self.assertAllClose(sess.run(out), 121)

      # Test caching using custom filesystem (file://) to make sure that the
      # TF Hub library can operate on such paths.
      try:
        root_dir = "file://%s" % self.get_temp_dir()
        cache_dir = "%s_%s" % (root_dir, "cache")
        tf.compat.v1.gfile.MakeDirs(cache_dir)
        os.environ["TFHUB_CACHE_DIR"] = cache_dir
        m = hub.Module("http://localhost:%d/test_module.tgz" % self.server_port)
        out = m(11)
        with tf.compat.v1.train.MonitoredSession() as sess:
          self.assertAllClose(sess.run(out), 121)

        cache_content = sorted(tf.compat.v1.gfile.ListDirectory(cache_dir))
        logging.info("Cache context: %s", str(cache_content))
        self.assertEqual(2, len(cache_content))
        self.assertTrue(cache_content[1].endswith(".descriptor.txt"))
        module_files = sorted(tf.compat.v1.gfile.ListDirectory(
            os.path.join(cache_dir, cache_content[0])))
        self.assertListEqual(
            ["assets", "saved_model.pb", "tfhub_module.pb", "variables"],
            module_files)
      finally:
        os.unsetenv("TFHUB_CACHE_DIR")

  def test_module_export_vocab_on_custom_fs(self):
    root_dir = "file://%s" % self.get_temp_dir()
    export_dir = "%s_%s" % (root_dir, "export")
    tf.compat.v1.gfile.MakeDirs(export_dir)
    # Create a module with a vocab file located on a custom filesystem.
    vocab_dir = os.path.join(root_dir, "vocab_location")
    tf.compat.v1.gfile.MakeDirs(vocab_dir)
    vocab_filename = os.path.join(vocab_dir, "tokens.txt")
    tf_utils.atomic_write_string_to_file(vocab_filename, "one", False)

    def create_assets_module_fn():

      def assets_module_fn():
        indices = tf.compat.v1.placeholder(dtype=tf.int64, name="indices")
        table = index_to_string_table_from_file(
            vocabulary_file=vocab_filename, default_value="UNKNOWN")
        outputs = table.lookup(indices)
        hub.add_signature(inputs=indices, outputs=outputs)

      return assets_module_fn

    with tf.Graph().as_default():
      assets_module_fn = create_assets_module_fn()
      spec = hub.create_module_spec(assets_module_fn)
      embedding_module = hub.Module(spec)
      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.tables_initializer())
        embedding_module.export(export_dir, sess)

    module_files = tf.compat.v1.gfile.ListDirectory(export_dir)
    self.assertListEqual(
        ["assets", "saved_model.pb", "tfhub_module.pb", "variables"],
        sorted(module_files))
    module_files = tf.compat.v1.gfile.ListDirectory(os.path.join(export_dir,
                                                                 "assets"))
    self.assertListEqual(["tokens.txt"], module_files)

  def test_resolve(self):
    with tf.Graph().as_default():
      self._generate_module()

      module_dir = hub.resolve(
          "http://localhost:%d/test_module.tgz" % self.server_port)
      self.assertIn(tempfile.gettempdir(), module_dir)
      module_files = sorted(tf.compat.v1.gfile.ListDirectory(module_dir))
      self.assertEqual(
          ["assets", "saved_model.pb", "tfhub_module.pb", "variables"],
          module_files)

  def test_load(self):
    if not hasattr(tf.compat.v1.saved_model, "load_v2"):
      try:
        hub.load("@my/tf2_module/2")
        self.fail("Failure expected. hub.load() not supported in TF 1.x")
      except NotImplementedError:
        pass
    elif tf.compat.v1.executing_eagerly():

      class AdderModule(tf.train.Checkpoint):

        @tf.function(
            input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
        def add(self, x):
          return x + x + 1.

      to_export = AdderModule()
      save_dir = os.path.join(self.get_temp_dir(), "saved_model_v2")
      tf.saved_model.save(to_export, save_dir)
      module_name = "test_module_v2.tgz"
      self._create_tgz(save_dir, module_name)

      restored_module = hub.load(
          "http://localhost:%d/%s" % (self.server_port, module_name))
      self.assertIsNotNone(restored_module)
      self.assertTrue(hasattr(restored_module, "add"))

  def test_load_v1(self):
    if (not hasattr(tf.compat.v1.saved_model, "load_v2") or
        not tf.compat.v1.executing_eagerly()):
      return  # The test only applies when running V2 mode.
    full_module_path = test_utils.get_test_data_path("half_plus_two_v1.tar.gz")
    os.chdir(os.path.dirname(full_module_path))
    server_port = test_utils.start_http_server()
    handle = "http://localhost:%d/half_plus_two_v1.tar.gz" % server_port
    hub.load(handle)


if __name__ == "__main__":
  tf.test.main()
