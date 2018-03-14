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
"""Tests for tensorflow_hub.compressed_module_resolver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint:disable=g-import-not-at-top,g-statement-before-imports
try:
  import mock as mock
except ImportError:
  import unittest.mock as mock
# pylint:disable=g-import-not-at-top,g-statement-before-imports

import os
import re
import socket
import tarfile
import tempfile
import uuid

import tensorflow as tf

from tensorflow_hub import compressed_module_resolver
from tensorflow_hub import resolver
from tensorflow_hub import test_utils
from tensorflow_hub import tf_utils

FLAGS = tf.flags.FLAGS


class HttpCompressedFileResolverTest(tf.test.TestCase):

  def setUp(self):
    # Set current directory to test temp directory where we can create
    # files and serve them through the HTTP server.
    os.chdir(self.get_temp_dir())

    # Create three temp files.
    self.files = ["file1", "file2", "file3"]
    for cur_file in self.files:
      with tf.gfile.GFile(cur_file, mode="w") as f:
        f.write(cur_file)

    # Create TAR files.
    tar = tarfile.open("mock_module.tar", "w")
    for name in self.files:
      tar.add(name)
    tar.close()

    # Create TGZ file
    tar = tarfile.open("mock_module.tar.gz", "w:gz")
    for name in self.files:
      tar.add(name)
    tar.close()

    self.server_port = test_utils.start_http_server()
    self.module_handle = (
        "http://localhost:%d/mock_module.tar.gz" % self.server_port)
    self.redirect_server_port = test_utils.start_http_server(
        "http://localhost:%d" % self.server_port)

  def testGetModulePathTar(self):
    FLAGS.tfhub_cache_dir = os.path.join(self.get_temp_dir(), "cache_dir")
    http_resolver = compressed_module_resolver.HttpCompressedFileResolver()
    path = http_resolver.get_module_path(
        "http://localhost:%d/mock_module.tar" %
        self.server_port)
    files = os.listdir(path)
    self.assertListEqual(sorted(files), ["file1", "file2", "file3"])

  def testGetModulePathTarGz(self):
    FLAGS.tfhub_cache_dir = os.path.join(self.get_temp_dir(), "cache_dir")
    http_resolver = compressed_module_resolver.HttpCompressedFileResolver()
    path = http_resolver.get_module_path(self.module_handle)
    files = os.listdir(path)
    self.assertListEqual(sorted(files), ["file1", "file2", "file3"])

  def testModuleDescriptor(self):
    FLAGS.tfhub_cache_dir = os.path.join(self.get_temp_dir(), "cache_dir")
    http_resolver = compressed_module_resolver.HttpCompressedFileResolver()
    path = http_resolver.get_module_path(self.module_handle)
    desc = tf_utils.read_file_to_string(resolver._module_descriptor_file(path))
    self.assertRegexpMatches(desc, "Module: %s\n"
                             "Download Time: .*\n"
                             "Downloader Hostname: %s .PID:%d." %
                             (re.escape(self.module_handle),
                              re.escape(socket.gethostname()), os.getpid()))

  def testNoCacheDirSet(self):
    FLAGS.tfhub_cache_dir = ""
    http_resolver = compressed_module_resolver.HttpCompressedFileResolver()
    handle = "http://localhost:%d/mock_module.tar.gz" % self.server_port
    path = http_resolver.get_module_path(handle)
    files = os.listdir(path)
    self.assertListEqual(sorted(files), ["file1", "file2", "file3"])
    self.assertStartsWith(path, tempfile.gettempdir())

  def testIsTarFile(self):
    self.assertTrue(compressed_module_resolver._is_tarfile("foo.tar"))
    self.assertTrue(compressed_module_resolver._is_tarfile("foo.tar.gz"))
    self.assertTrue(compressed_module_resolver._is_tarfile("foo.tgz"))
    self.assertFalse(compressed_module_resolver._is_tarfile("foo"))
    self.assertFalse(compressed_module_resolver._is_tarfile("footar"))

  def testAbandondedLockFile(self):
    # Tests that the caching procedure is resilient to an abandonded lock
    # file.
    FLAGS.tfhub_cache_dir = os.path.join(self.get_temp_dir(), "cache_dir")

    # Create an "abandoned" lock file, i.e. a lock file with no process actively
    # downloading anymore.
    module_dir = compressed_module_resolver._module_dir(FLAGS.tfhub_cache_dir,
                                                        self.module_handle)
    task_uid = uuid.uuid4().hex
    lock_filename = resolver._lock_filename(module_dir)
    tf_utils.atomic_write_string_to_file(lock_filename,
                                         resolver._lock_file_contents(task_uid),
                                         overwrite=False)
    with mock.patch.object(
        compressed_module_resolver.HttpCompressedFileResolver,
        "_lock_file_timeout_sec",
        return_value=10):
      http_resolver = compressed_module_resolver.HttpCompressedFileResolver()
      handle = "http://localhost:%d/mock_module.tar.gz" % self.server_port
      # After seeing the lock file is abandoned, this resolver will download the
      # module and return a path to the extracted contents.
      path = http_resolver.get_module_path(handle)
    files = os.listdir(path)
    self.assertListEqual(sorted(files), ["file1", "file2", "file3"])
    self.assertFalse(tf.gfile.Exists(lock_filename))

  def testModuleAlreadyDownloaded(self):
    FLAGS.tfhub_cache_dir = os.path.join(self.get_temp_dir(), "cache_dir")
    http_resolver = compressed_module_resolver.HttpCompressedFileResolver()
    path = http_resolver.get_module_path(self.module_handle)
    files = sorted(os.listdir(path))
    self.assertListEqual(files, ["file1", "file2", "file3"])
    creation_times = [
        tf.gfile.Stat(os.path.join(path, f)).mtime_nsec for f in files
    ]
    # Call resolver again and make sure that the module is not downloaded again
    # by checking the timestamps of the module files.
    path = http_resolver.get_module_path(self.module_handle)
    files = sorted(os.listdir(path))
    self.assertListEqual(files, ["file1", "file2", "file3"])
    self.assertListEqual(
        creation_times,
        [tf.gfile.Stat(os.path.join(path, f)).mtime_nsec for f in files])

  def testCorruptedArchive(self):
    with tf.gfile.GFile("bad_archive.tar.gz", mode="w") as f:
      f.write("bad_archive")
    http_resolver = compressed_module_resolver.HttpCompressedFileResolver()
    try:
      http_resolver.get_module_path(
          "http://localhost:%d/bad_archive.tar.gz" % self.server_port)
      self.fail("Corrupted archive should have failed to resolve.")
    except IOError as e:
      self.assertEqual(
          "http://localhost:%d/bad_archive.tar.gz is not a valid tar archive." %
          self.server_port, str(e))
    try:
      http_resolver.get_module_path(
          "http://localhost:%d/bad_archive.tar.gz" % self.redirect_server_port)
      self.fail("Corrupted archive should have failed to resolve.")
    except IOError as e:
      # Check that the error message contain the ultimate (redirected to) URL.
      self.assertEqual(
          "http://localhost:%d/bad_archive.tar.gz is not a valid tar archive." %
          self.redirect_server_port, str(e))


if __name__ == "__main__":
  tf.test.main()
