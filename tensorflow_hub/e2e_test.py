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

from absl import logging
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow_hub import test_utils
from tensorflow_hub import tf_utils
from tensorflow_hub import tf_v1

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops.lookup_ops import index_to_string_table_from_file
# pylint: enable=g-direct-tensorflow-import


class End2EndTest(tf.test.TestCase):

  def setUp(self):
    # Set current directory to test temp directory where we can create
    # files and serve them through the HTTP server.
    os.chdir(self.get_temp_dir())

    self.server_port = test_utils.start_http_server()

  def _stateless_module_fn(self):
    """Simple module that squares an input."""
    x = tf_v1.placeholder(tf.int64)
    y = x*x
    hub.add_signature(inputs=x, outputs=y)

  def _list_module_files(self, module_dir):
    files = []
    for f in tf_v1.gfile.ListDirectory(module_dir):
      full_path = os.path.join(module_dir, f)
      stat_res = tf_v1.gfile.Stat(full_path)
      if stat_res.is_directory:
        files.extend(self._list_module_files(full_path))
      else:
        files.append(f)
    return files

  def test_http_locations(self):
    with tf.Graph().as_default():
      spec = hub.create_module_spec(self._stateless_module_fn)
      m = hub.Module(spec, name="test_module")
      out = m(10)

      export_path = os.path.join(self.get_temp_dir(), "module")
      with tf_v1.Session() as sess:
        sess.run(tf_v1.global_variables_initializer())
        self.assertAllClose(sess.run(out), 100)
        m.export(export_path, sess)

      os.chdir(export_path)

      tar = tarfile.open("test_module.tgz", "w")
      for f in self._list_module_files(export_path):
        tar.add(f)
      tar.close()

      m = hub.Module("http://localhost:%d/test_module.tgz" % self.server_port)
      out = m(11)
      with tf_v1.Session() as sess:
        self.assertAllClose(sess.run(out), 121)

      # Test caching using custom filesystem (file://) to make sure that the
      # TF Hub library can operate on such paths.
      try:
        root_dir = "file://%s" % self.get_temp_dir()
        cache_dir = "%s_%s" % (root_dir, "cache")
        tf_v1.gfile.MakeDirs(cache_dir)
        os.environ["TFHUB_CACHE_DIR"] = cache_dir
        m = hub.Module("http://localhost:%d/test_module.tgz" % self.server_port)
        out = m(11)
        with tf_v1.train.MonitoredSession() as sess:
          self.assertAllClose(sess.run(out), 121)

        cache_content = sorted(tf_v1.gfile.ListDirectory(cache_dir))
        logging.info("Cache context: %s", str(cache_content))
        self.assertEqual(2, len(cache_content))
        self.assertTrue(cache_content[1].endswith(".descriptor.txt"))
        module_files = sorted(tf_v1.gfile.ListDirectory(
            os.path.join(cache_dir, cache_content[0])))
        self.assertListEqual(["saved_model.pb", "tfhub_module.pb"],
                             module_files)
      finally:
        os.unsetenv("TFHUB_CACHE_DIR")

  def test_module_export_vocab_on_custom_fs(self):
    root_dir = "file://%s" % self.get_temp_dir()
    export_dir = "%s_%s" % (root_dir, "export")
    tf_v1.gfile.MakeDirs(export_dir)
    # Create a module with a vocab file located on a custom filesystem.
    vocab_dir = os.path.join(root_dir, "vocab_location")
    tf_v1.gfile.MakeDirs(vocab_dir)
    vocab_filename = os.path.join(vocab_dir, "tokens.txt")
    tf_utils.atomic_write_string_to_file(vocab_filename, "one", False)

    def create_assets_module_fn():

      def assets_module_fn():
        indices = tf_v1.placeholder(dtype=tf.int64, name="indices")
        table = index_to_string_table_from_file(
            vocabulary_file=vocab_filename, default_value="UNKNOWN")
        outputs = table.lookup(indices)
        hub.add_signature(inputs=indices, outputs=outputs)

      return assets_module_fn

    with tf.Graph().as_default():
      assets_module_fn = create_assets_module_fn()
      spec = hub.create_module_spec(assets_module_fn)
      embedding_module = hub.Module(spec)
      with tf_v1.Session() as sess:
        sess.run(tf_v1.tables_initializer())
        embedding_module.export(export_dir, sess)

    module_files = tf_v1.gfile.ListDirectory(export_dir)
    self.assertListEqual(
        ["assets", "saved_model.pb", "tfhub_module.pb", "variables"],
        sorted(module_files))
    module_files = tf_v1.gfile.ListDirectory(os.path.join(export_dir, "assets"))
    self.assertListEqual(["tokens.txt"], module_files)

if __name__ == "__main__":
  tf.test.main()
