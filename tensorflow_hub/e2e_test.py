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

import os
import tarfile

from absl import logging
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow_hub import test_utils


class End2EndTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
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
    module_export_path = os.path.join(self.get_temp_dir(), "module")
    test_utils.export_module(module_export_path)
    self._create_tgz(module_export_path)

  def test_http_locations(self):
    self._generate_module()

    m = hub.load("http://localhost:%d/test_module.tgz" % self.server_port)
    self.assertAllClose(m(11), 121)

    # Test caching using custom filesystem (file://) to make sure that the
    # TF Hub library can operate on such paths.
    try:
      root_dir = "file://%s" % self.get_temp_dir()
      cache_dir = "%s_%s" % (root_dir, "cache")
      tf.compat.v1.gfile.MakeDirs(cache_dir)
      os.environ["TFHUB_CACHE_DIR"] = cache_dir
      m = hub.load("http://localhost:%d/test_module.tgz" % self.server_port)
      self.assertAllClose(m(11), 121)

      cache_content = sorted(tf.compat.v1.gfile.ListDirectory(cache_dir))
      logging.info("Cache context: %s", str(cache_content))
      self.assertEqual(2, len(cache_content))
      self.assertTrue(cache_content[1].endswith(".descriptor.txt"))
      module_files = sorted(
          tf.compat.v1.gfile.ListDirectory(
              os.path.join(cache_dir, cache_content[0])
          )
      )
      self.assertListEqual(
          ["assets", "fingerprint.pb", "saved_model.pb", "variables"],
          module_files,
      )
    finally:
      os.unsetenv("TFHUB_CACHE_DIR")

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
