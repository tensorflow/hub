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
"""Tests for tensorflow_hub.resolver."""

import os
import re
import socket
import tempfile
import threading
import time
import unittest
from unittest import mock
import uuid

from absl import flags
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_hub import compressed_module_resolver
from tensorflow_hub import config
from tensorflow_hub import registry
from tensorflow_hub import resolver
from tensorflow_hub import test_utils
from tensorflow_hub import tf_utils
from tensorflow_hub import uncompressed_module_resolver

FLAGS = flags.FLAGS


class PathResolverTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.resolver = resolver.PathResolver()

  def testAlwaysSupported(self):
    os.chdir(os.path.join(self.get_temp_dir()))
    self.assertTrue(self.resolver.is_supported("/tmp"))
    tf.compat.v1.gfile.MkDir("foo/")
    self.assertTrue(self.resolver.is_supported("./foo/"))
    self.assertTrue(self.resolver.is_supported("foo/"))
    # Directory doesn't exist.
    self.assertTrue(self.resolver.is_supported("bar/"))
    self.assertTrue(self.resolver.is_supported("foo/bar"))
    self.assertTrue(self.resolver.is_supported("nope://throw-OpError"))

  def testCallWithValidHandle(self):
    tmp_path = os.path.join(self.get_temp_dir(), "1234")
    tf.compat.v1.gfile.MkDir(tmp_path)
    path = self.resolver(tmp_path)
    self.assertEqual(path, tmp_path)

  def testCallWhenHandleDirectoryDoesNotExist(self):
    with self.assertRaisesRegex(IOError, "/foo/ does not exist."):
      self.resolver("/foo/")


class FakeResolver(resolver.Resolver):
  """Fake Resolver used to test composite Resolvers."""

  def __init__(self, prefix):
    self.prefix = prefix

  def is_supported(self, handle):
    return handle.startswith(self.prefix)

  def __call__(self, handle):
    if handle.endswith("error"):
      raise ValueError("error for: " + handle)
    return handle + "-resolved_by_" + self.prefix


class CompressedResolverTest(tf.test.TestCase):

  def testCacheDir(self):
    # No cache dir set, None is returned.
    cache_dir = resolver.tfhub_cache_dir()
    self.assertEqual(cache_dir, None)
    # Use temp dir.
    cache_dir = resolver.tfhub_cache_dir(use_temp=True)
    self.assertEquals(cache_dir,
                      os.path.join(tempfile.gettempdir(), "tfhub_modules"))
    # Use override
    cache_dir = resolver.tfhub_cache_dir(default_cache_dir="/d", use_temp=True)
    self.assertEqual("/d", cache_dir)
    # Use a flag
    FLAGS.tfhub_cache_dir = "/e"
    cache_dir = resolver.tfhub_cache_dir(default_cache_dir="/d", use_temp=True)
    self.assertEqual("/e", cache_dir)
    FLAGS.tfhub_cache_dir = ""
    # Use env variable
    os.environ[resolver._TFHUB_CACHE_DIR] = "/f"
    cache_dir = resolver.tfhub_cache_dir(default_cache_dir="/d", use_temp=True)
    self.assertEqual("/f", cache_dir)
    FLAGS.tfhub_cache_dir = "/e"
    cache_dir = resolver.tfhub_cache_dir(default_cache_dir="/d", use_temp=True)
    self.assertEqual("/f", cache_dir)
    FLAGS.tfhub_cache_dir = ""
    os.unsetenv(resolver._TFHUB_CACHE_DIR)

  def testDirSize(self):
    fake_task_uid = 1234

    # Create a directory with some files and sub-directory and check its size.
    test_dir = resolver._temp_download_dir(self.get_temp_dir(), fake_task_uid)
    tf.compat.v1.gfile.MakeDirs(test_dir)
    tf_utils.atomic_write_string_to_file(
        os.path.join(test_dir, "file1"), "content1", False)
    tf_utils.atomic_write_string_to_file(
        os.path.join(test_dir, "file2"), "content2", False)
    test_sub_dir = os.path.join(test_dir, "sub_dir")
    tf.compat.v1.gfile.MakeDirs(test_sub_dir)
    tf_utils.atomic_write_string_to_file(
        os.path.join(test_sub_dir, "file3"), "content3", False)
    self.assertEqual(3 * 8, resolver._dir_size(test_dir))
    self.assertEqual(8, resolver._dir_size(test_sub_dir))

    # Treat the directory as a temporary directory used by a module download by
    # referring to that directory from the lock file.
    fake_lock_filename = resolver._lock_filename(self.get_temp_dir())
    tf_utils.atomic_write_string_to_file(
        fake_lock_filename, resolver._lock_file_contents(fake_task_uid), False)
    self.assertEqual(3 * 8, resolver._locked_tmp_dir_size(fake_lock_filename))

    # Check that if temp directory doesn't exist, 0 is returned.
    tf.compat.v1.gfile.DeleteRecursively(test_dir)
    self.assertEqual(0, resolver._locked_tmp_dir_size(fake_lock_filename))

  def testLockFileName(self):
    self.assertEquals("/a/b/c.lock", resolver._lock_filename("/a/b/c/"))

  def testTempDownloadDir(self):
    self.assertEquals("/a/b.t.tmp", resolver._temp_download_dir("/a/b/", "t"))

  def testReadTaskUidFromLockFile(self):
    module_dir = os.path.join(self.get_temp_dir(), "module")
    task_uid = uuid.uuid4().hex
    lock_filename = resolver._lock_filename(module_dir)
    tf_utils.atomic_write_string_to_file(
        lock_filename, resolver._lock_file_contents(task_uid), overwrite=False)
    self.assertEqual(task_uid, resolver._task_uid_from_lock_file(lock_filename))

  def testWaitForLockToDisappear_DownloadCompletes(self):
    module_dir = os.path.join(self.get_temp_dir(), "module")
    task_uid = uuid.uuid4().hex
    lock_filename = resolver._lock_filename(module_dir)
    # Write lock file
    tf_utils.atomic_write_string_to_file(
        lock_filename, resolver._lock_file_contents(task_uid), overwrite=False)
    # Wait for the lock file to disappear (in a separate thread)
    thread = threading.Thread(
        target=resolver._wait_for_lock_to_disappear,
        args=(
            "module",
            lock_filename,
            600,
        ))
    thread.start()
    # Delete the lock file.
    tf.compat.v1.gfile.Remove(lock_filename)
    thread.join(10)
    # The waiting terminates without errors.

  def testWaitForLockToDisappear_DownloadOngoing(self):
    module_dir = os.path.join(self.get_temp_dir(), "module")
    task_uid = uuid.uuid4().hex
    lock_filename = resolver._lock_filename(module_dir)
    lock_file_content = resolver._lock_file_contents(task_uid)
    tf_utils.atomic_write_string_to_file(
        lock_filename, lock_file_content, overwrite=False)

    lock_expiration_wait_time_secs = 10
    thread = threading.Thread(
        target=resolver._wait_for_lock_to_disappear,
        args=(
            "module",
            lock_filename,
            lock_expiration_wait_time_secs,
        ))
    thread.start()
    # Simulate download by writing a file every 1 sec. While writes are happing
    # the lock file remains in place.
    tmp_dir = resolver._temp_download_dir(self.get_temp_dir(), task_uid)
    tf.compat.v1.gfile.MakeDirs(tmp_dir)
    for x in range(2 * lock_expiration_wait_time_secs):
      tf_utils.atomic_write_string_to_file(
          os.path.join(tmp_dir, "file_%d" % x), "test", overwrite=False)
      # While writes are happening the original lock file is in place.
      self.assertEqual(lock_file_content,
                       tf_utils.read_file_to_string(lock_filename))
      time.sleep(1)
    thread.join(lock_expiration_wait_time_secs)
    # The waiting terminates without errors.

  def testWaitForLockToDisappear_DownloadAborted(self):
    module_dir = os.path.join(self.get_temp_dir(), "module")
    task_uid = uuid.uuid4().hex
    lock_filename = resolver._lock_filename(module_dir)
    lock_file_content = resolver._lock_file_contents(task_uid)
    tf_utils.atomic_write_string_to_file(
        lock_filename, lock_file_content, overwrite=False)
    tmp_dir = resolver._temp_download_dir(self.get_temp_dir(), task_uid)
    tf.compat.v1.gfile.MakeDirs(tmp_dir)

    thread = threading.Thread(
        target=resolver._wait_for_lock_to_disappear,
        args=(
            "module",
            lock_filename,
            10,
        ))
    thread.start()
    thread.join(30)
    # Because nobody was writing to tmp_dir, the lock file got reclaimed by
    # resolver._wait_for_lock_to_disappear.
    self.assertFalse(tf.compat.v1.gfile.Exists(lock_filename))

  def testModuleAlreadyDownloaded(self):
    # Simulate the case when a rogue process finishes downloading a module
    # right before the current process can perform a rename of a temp directory
    # to a permanent module directory.
    module_dir = os.path.join(self.get_temp_dir(), "module")

    def fake_download_fn_with_rogue_behavior(handle, tmp_dir):
      del handle, tmp_dir
      # Create module directory
      tf.compat.v1.gfile.MakeDirs(module_dir)
      tf_utils.atomic_write_string_to_file(
          os.path.join(module_dir, "file"), "content", False)

    self.assertEqual(
        module_dir,
        resolver.atomic_download("module", fake_download_fn_with_rogue_behavior,
                                 module_dir))
    self.assertEqual(tf.compat.v1.gfile.ListDirectory(module_dir), ["file"])
    self.assertFalse(
        tf.compat.v1.gfile.Exists(resolver._lock_filename(module_dir)))
    parent_dir = os.path.abspath(os.path.join(module_dir, ".."))
    self.assertEqual(
        sorted(tf.compat.v1.gfile.ListDirectory(parent_dir)),
        ["module", "module.descriptor.txt"])
    self.assertRegexpMatches(
        tf_utils.read_file_to_string(
            resolver._module_descriptor_file(module_dir)), "Module: module\n"
        "Download Time: .*\n"
        "Downloader Hostname: %s .PID:%d." %
        (re.escape(socket.gethostname()), os.getpid()))

    # Try downloading the model again. Mock
    # tf_utils.atomic_write_string_to_file() to throw an exception. Since the
    # model is already downloaded, the function will never get called and the
    # download succeeds.
    with mock.patch.object(
        tf_utils,
        "atomic_write_string_to_file",
        side_effect=ValueError("This error should never be raised!")):
      self.assertEqual(
          module_dir,
          resolver.atomic_download("module",
                                   fake_download_fn_with_rogue_behavior,
                                   module_dir))
      self.assertEqual(tf.compat.v1.gfile.ListDirectory(module_dir), ["file"])
      self.assertFalse(
          tf.compat.v1.gfile.Exists(resolver._lock_filename(module_dir)))

  def testModuleDownloadedWhenEmptyFolderExists(self):
    # Simulate the case when a module is cached in /tmp/module_dir but module
    # files inside the folder are deleted. In this case, the download should
    # still be conducted.
    module_dir = os.path.join(self.get_temp_dir(), "module")

    def fake_download_fn(handle, tmp_dir):
      del handle, tmp_dir
      tf.compat.v1.gfile.MakeDirs(module_dir)
      tf_utils.atomic_write_string_to_file(
          os.path.join(module_dir, "file"), "content", False)

    # Create an empty folder before downloading.
    self.assertFalse(tf.compat.v1.gfile.Exists(module_dir))
    tf.compat.v1.gfile.MakeDirs(module_dir)

    self.assertEqual(
        module_dir,
        resolver.atomic_download("module", fake_download_fn, module_dir))
    self.assertEqual(tf.compat.v1.gfile.ListDirectory(module_dir), ["file"])
    self.assertFalse(
        tf.compat.v1.gfile.Exists(resolver._lock_filename(module_dir)))
    parent_dir = os.path.abspath(os.path.join(module_dir, ".."))
    self.assertEqual(
        sorted(tf.compat.v1.gfile.ListDirectory(parent_dir)),
        ["module", "module.descriptor.txt"])
    self.assertRegexpMatches(
        tf_utils.read_file_to_string(
            resolver._module_descriptor_file(module_dir)), "Module: module\n"
        "Download Time: .*\n"
        "Downloader Hostname: %s .PID:%d." %
        (re.escape(socket.gethostname()), os.getpid()))

  def testModuleConcurrentDownload(self):
    module_dir = os.path.join(self.get_temp_dir(), "module")

    # To simulate one downloading starting while the other is still in progress,
    # call resolver.atomic_download() from download_fn(). The second download
    # is set up with download_fn() that fails. That download_fn() is not
    # expected to be called.
    def second_download_fn(handle, tmp_dir):
      del handle, tmp_dir
      self.fail("This should not be called. The module should have been "
                "downloaded already.")

    second_download_thread = threading.Thread(
        target=resolver.atomic_download,
        args=(
            "module",
            second_download_fn,
            module_dir,
        ))

    def first_download_fn(handle, tmp_dir):
      del handle, tmp_dir
      tf.compat.v1.gfile.MakeDirs(module_dir)
      tf_utils.atomic_write_string_to_file(
          os.path.join(module_dir, "file"), "content", False)
      second_download_thread.start()

    self.assertEqual(
        module_dir,
        resolver.atomic_download("module", first_download_fn, module_dir))
    second_download_thread.join(30)
    # The waiting terminates without errors.

  # def testModuleDownloadPermissionDenied(self):
  #   readonly_dir = os.path.join(self.get_temp_dir(), "readonly")
  #   os.mkdir(readonly_dir, 0o500)
  #   module_dir = os.path.join(readonly_dir, "module")

  #   def unused_download_fn(handle, tmp_dir):
  #     del handle, tmp_dir
  #     self.fail("This should not be called. Already writing the lockfile "
  #               "is expected to raise an error.")

  #   with self.assertRaises(tf.errors.PermissionDeniedError):
  #     resolver.atomic_download("module", unused_download_fn, module_dir)

  def testModuleLockLostDownloadKilled(self):
    module_dir = os.path.join(self.get_temp_dir(), "module")
    download_aborted_msg = "Download aborted."

    def kill_download(handle, tmp_dir):
      del handle, tmp_dir
      # Simulate lock loss by removing the lock.
      tf.compat.v1.gfile.Remove(resolver._lock_filename(module_dir))
      # Throw an error to simulate aborted download.
      raise OSError(download_aborted_msg)

    try:
      resolver.atomic_download("module", kill_download, module_dir)
      self.fail("atomic_download() should have thrown an exception.")
    except OSError as _:
      pass
    parent_dir = os.path.abspath(os.path.join(module_dir, ".."))
    # Test that all files got cleaned up.
    self.assertEqual(tf.compat.v1.gfile.ListDirectory(parent_dir), [])

  def testNotFoundGCSBucket(self):
    # When trying to use not existing GCS bucket, test that
    # tf_util.atomic_write_string_to_file raises tf.error.NotFoundError.
    # Other errors that may arise from bad network connectivity are ignored by
    # resolver.atomic_download and retried infinitely.
    module_dir = ""

    def dummy_download_fn(handle, tmp_dir):
      del handle, tmp_dir
      return

    # Simulate missing GCS bucket by raising NotFoundError in
    # atomic_write_string_to_file.
    with unittest.mock.patch(
        "tensorflow_hub.tf_utils.atomic_write_string_to_file") as mock_:
      mock_.side_effect = tf.errors.NotFoundError(None, None, "Test")
      try:
        resolver.atomic_download("module", dummy_download_fn, module_dir)
        assert False
      except tf.errors.NotFoundError as e:
        self.assertEqual("Test", e.message)


class UncompressedResolverTest(tf.test.TestCase):

  def testModuleRunningWithUncompressedContext(self):
    module_export_path = os.path.join(self.get_temp_dir(), "module")
    with tf.Graph().as_default():
      test_utils.export_module(module_export_path)
      # Mock the server by returning the path to the local uncompressed module
      with mock.patch.object(
          uncompressed_module_resolver.HttpUncompressedFileResolver,
          "_request_gcs_location",
          return_value=module_export_path) as mocked_urlopen:
        with test_utils.UncompressedLoadFormatContext():
          m = hub.Module("https://tfhub.dev/google/model/1")
        mocked_urlopen.assert_called_once_with(
            "https://tfhub.dev/google/model/1?tf-hub-format=uncompressed")
      out = m(11)
      with tf.compat.v1.Session() as sess:
        self.assertAllClose(sess.run(out), 121)


class LoadFormatResolverBehaviorTest(tf.test.TestCase):
  """Test that the right resolvers are called depending on the load format."""

  def _assert_resolver_is_called(self, http_resolver):
    module_url = "https://tfhub.dev/google/model/1"
    with mock.patch.object(
        http_resolver, "__call__", side_effect=ValueError) as mocked_call:
      try:
        hub.Module(module_url)
        self.fail("Failure expected since mock raises it as side effect.")
      except ValueError:
        pass
    mocked_call.assert_called_once_with(module_url)

  def _assert_compressed_resolver_called(self):
    self._assert_resolver_is_called(
        compressed_module_resolver.HttpCompressedFileResolver)

  def _assert_uncompressed_resolver_called(self):
    self._assert_resolver_is_called(
        uncompressed_module_resolver.HttpUncompressedFileResolver)

  def test_load_format_auto(self):
    # ModelLoadFormat is set to AUTO on default
    self._assert_compressed_resolver_called()

  def test_load_format_compressed(self):
    with test_utils.CompressedLoadFormatContext():
      self._assert_compressed_resolver_called()

  def test_load_format_uncompressed(self):
    with test_utils.UncompressedLoadFormatContext():
      self._assert_uncompressed_resolver_called()


if __name__ == "__main__":
  # Make OSS configuration used for resolvers/loaders.
  registry._clear()
  config._run()
  tf.test.main()
