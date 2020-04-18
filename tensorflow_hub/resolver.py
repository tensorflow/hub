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
"""Interface and common utility methods to perform module address resolution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import datetime
import os
import socket
import sys
import tarfile
import tempfile
import time
import uuid

from absl import flags
from absl import logging
import tensorflow as tf
from tensorflow_hub import tf_utils
from tensorflow_hub import tf_v1


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "tfhub_cache_dir",
    None,
    "If set, TF-Hub will download and cache Modules into this directory. "
    "Otherwise it will attempt to find a network path.")

_TFHUB_CACHE_DIR = "TFHUB_CACHE_DIR"
_TFHUB_DOWNLOAD_PROGRESS = "TFHUB_DOWNLOAD_PROGRESS"


def tfhub_cache_dir(default_cache_dir=None, use_temp=False):
  """Returns cache directory.

  Returns cache directory from either TFHUB_CACHE_DIR environment variable
  or --tfhub_cache_dir or default, if set.

  Args:
    default_cache_dir: Default cache location to use if neither TFHUB_CACHE_DIR
                       environment variable nor --tfhub_cache_dir are
                       not specified.
    use_temp: bool, Optional to enable using system's temp directory as a
              module cache directory if neither default_cache_dir nor
              --tfhub_cache_dir nor TFHUB_CACHE_DIR environment variable are
              specified .
  """

  # Note: We are using FLAGS["tfhub_cache_dir"] (and not FLAGS.tfhub_cache_dir)
  # to access the flag value in order to avoid parsing argv list. The flags
  # should have been parsed by now in main() by tf.app.run(). If that was not
  # the case (say in Colab env) we skip flag parsing because argv may contain
  # unknown flags.
  cache_dir = (
      os.getenv(_TFHUB_CACHE_DIR, "") or FLAGS["tfhub_cache_dir"].value or
      default_cache_dir)
  if not cache_dir and use_temp:
    # Place all TF-Hub modules under <system's temp>/tfhub_modules.
    cache_dir = os.path.join(tempfile.gettempdir(), "tfhub_modules")
  if cache_dir:
    logging.log_first_n(logging.INFO, "Using %s to cache modules.", 1,
                        cache_dir)
  return cache_dir


def create_local_module_dir(cache_dir, module_name):
  """Creates and returns the name of directory where to cache a module."""
  tf_v1.gfile.MakeDirs(cache_dir)
  return os.path.join(cache_dir, module_name)


class DownloadManager(object):
  """Helper class responsible for TF-Hub module download and extraction."""

  def __init__(self, url):
    """Creates DownloadManager responsible for downloading a TF-Hub module.

    Args:
       url: URL pointing to the TF-Hub module to download and extract.
    """
    self._url = url
    self._last_progress_msg_print_time = time.time()
    self._total_bytes_downloaded = 0
    self._max_prog_str = 0

  def _print_download_progress_msg(self, msg, flush=False):
    """Prints a message about download progress either to the console or TF log.

    Args:
      msg: Message to print.
      flush: Indicates whether to flush the output (only used in interactive
             mode).
    """
    if self._interactive_mode():
      # Print progress message to console overwriting previous progress
      # message.
      self._max_prog_str = max(self._max_prog_str, len(msg))
      sys.stdout.write("\r%-{}s".format(self._max_prog_str) % msg)
      sys.stdout.flush()
      if flush:
        print("\n")
    else:
      # Interactive progress tracking is disabled. Print progress to the
      # standard TF log.
      logging.info(msg)

  def _log_progress(self, bytes_downloaded):
    """Logs progress information about ongoing module download.

    Args:
      bytes_downloaded: Number of bytes downloaded.
    """
    self._total_bytes_downloaded += bytes_downloaded
    now = time.time()
    if (self._interactive_mode() or
        now - self._last_progress_msg_print_time > 15):
      # Print progress message every 15 secs or if interactive progress
      # tracking is enabled.
      self._print_download_progress_msg(
          "Downloading %s: %s" % (self._url,
                                  tf_utils.bytes_to_readable_str(
                                      self._total_bytes_downloaded, True)))
      self._last_progress_msg_print_time = now

  def _interactive_mode(self):
    """Returns true if interactive logging is enabled."""
    return os.getenv(_TFHUB_DOWNLOAD_PROGRESS, "")

  def _extract_file(self, tgz, tarinfo, dst_path, buffer_size=10<<20):
    """Extracts 'tarinfo' from 'tgz' and writes to 'dst_path'."""
    src = tgz.extractfile(tarinfo)
    dst = tf_v1.gfile.GFile(dst_path, "wb")
    while 1:
      buf = src.read(buffer_size)
      if not buf:
        break
      dst.write(buf)
      self._log_progress(len(buf))
    dst.close()
    src.close()

  def download_and_uncompress(self, fileobj, dst_path):
    """Streams the content for the 'fileobj' and stores the result in dst_path.

    Args:
      fileobj: File handle pointing to .tar/.tar.gz content.
      dst_path: Absolute path where to store uncompressed data from 'fileobj'.

    Raises:
      ValueError: Unknown object encountered inside the TAR file.
    """
    try:
      with tarfile.open(mode="r|*", fileobj=fileobj) as tgz:
        for tarinfo in tgz:
          abs_target_path = _merge_relative_path(dst_path, tarinfo.name)

          if tarinfo.isfile():
            self._extract_file(tgz, tarinfo, abs_target_path)
          elif tarinfo.isdir():
            tf_v1.gfile.MakeDirs(abs_target_path)
          else:
            # We do not support symlinks and other uncommon objects.
            raise ValueError(
                "Unexpected object type in tar archive: %s" % tarinfo.type)

        total_size_str = tf_utils.bytes_to_readable_str(
            self._total_bytes_downloaded, True)
        self._print_download_progress_msg(
            "Downloaded %s, Total size: %s" % (self._url, total_size_str),
            flush=True)
    except tarfile.ReadError:
      raise IOError("%s does not appear to be a valid module." % self._url)


def _merge_relative_path(dst_path, rel_path):
  """Merge a relative tar file to a destination (which can be "gs://...")."""
  # Convert rel_path to be relative and normalize it to remove ".", "..", "//",
  # which are valid directories in fileystems like "gs://".
  norm_rel_path = os.path.normpath(rel_path.lstrip("/"))

  if norm_rel_path == ".":
    return dst_path

  # Check that the norm rel path does not starts with "..".
  if norm_rel_path.startswith(".."):
    raise ValueError("Relative path %r is invalid." % rel_path)

  merged = os.path.join(dst_path, norm_rel_path)

  # After merging verify that the merged path keeps the original dst_path.
  if not merged.startswith(dst_path):
    raise ValueError("Relative path %r is invalid. Failed to merge with %r." % (
        rel_path, dst_path))
  return merged


def _module_descriptor_file(module_dir):
  """Returns the name of the file containing descriptor for the 'module_dir'."""
  return "{}.descriptor.txt".format(module_dir)


def _write_module_descriptor_file(handle, module_dir):
  """Writes a descriptor file about the directory containing a module.

  Args:
    handle: Module name/handle.
    module_dir: Directory where a module was downloaded.
  """
  readme = _module_descriptor_file(module_dir)
  readme_content = (
      "Module: %s\nDownload Time: %s\nDownloader Hostname: %s (PID:%d)" %
      (handle, str(datetime.datetime.today()), socket.gethostname(),
       os.getpid()))
  # The descriptor file has no semantic meaning so we allow 'overwrite' since
  # there is a chance that another process might have written the file (and
  # crashed), we just overwrite it.
  tf_utils.atomic_write_string_to_file(readme, readme_content, overwrite=True)


def _lock_file_contents(task_uid):
  """Returns the content of the lock file."""
  return "%s.%d.%s" % (socket.gethostname(), os.getpid(), task_uid)


def _lock_filename(module_dir):
  """Returns lock file name."""
  return tf_utils.absolute_path(module_dir) + ".lock"


def _module_dir(lock_filename):
  """Returns module dir from a full 'lock_filename' path.

  Args:
    lock_filename: Name of the lock file, ends with .lock.

  Raises:
    ValueError: if lock_filename is ill specified.
  """
  if not lock_filename.endswith(".lock"):
    raise ValueError(
        "Lock file name (%s) has to end with .lock." % lock_filename)
  return lock_filename[0:-len(".lock")]


def _task_uid_from_lock_file(lock_filename):
  """Returns task UID of the task that created a given lock file."""
  lock = tf_utils.read_file_to_string(lock_filename)
  return lock.split(".")[-1]


def _temp_download_dir(module_dir, task_uid):
  """Returns the name of a temporary directory to download module to."""
  return "{}.{}.tmp".format(tf_utils.absolute_path(module_dir), task_uid)


def _dir_size(directory):
  """Returns total size (in bytes) of the given 'directory'."""
  size = 0
  for elem in tf_v1.gfile.ListDirectory(directory):
    elem_full_path = os.path.join(directory, elem)
    stat = tf_v1.gfile.Stat(elem_full_path)
    size += _dir_size(elem_full_path) if stat.is_directory else stat.length
  return size


def _locked_tmp_dir_size(lock_filename):
  """Returns the size of the temp dir pointed to by the given lock file."""
  task_uid = _task_uid_from_lock_file(lock_filename)
  try:
    return _dir_size(
        _temp_download_dir(_module_dir(lock_filename), task_uid))
  except tf.errors.NotFoundError:
    return 0


def _wait_for_lock_to_disappear(handle, lock_file, lock_file_timeout_sec):
  """Waits for the lock file to disappear.

  The lock file was created by another process that is performing a download
  into its own temporary directory. The name of this temp directory is
  sha1(<module>).<uuid>.tmp where <uuid> comes from the lock file.

  Args:
    handle: The location from where a module is being download.
    lock_file: Lock file created by another process downloading this module.
    lock_file_timeout_sec: The amount of time to wait (in seconds) before we
                           can declare that the other downloaded has been
                           abandoned. The download is declared abandoned if
                           there is no file size change in the temporary
                           directory within the last 'lock_file_timeout_sec'.
  """
  locked_tmp_dir_size = 0
  locked_tmp_dir_size_check_time = time.time()
  lock_file_content = None
  while tf_v1.gfile.Exists(lock_file):
    try:
      logging.log_every_n(
          logging.INFO,
          "Module '%s' already being downloaded by '%s'. Waiting.", 10,
          handle, tf_utils.read_file_to_string(lock_file))
      if (time.time() - locked_tmp_dir_size_check_time >
          lock_file_timeout_sec):
        # Check whether the holder of the current lock downloaded anything
        # in its temporary directory in the last 'lock_file_timeout_sec'.
        cur_locked_tmp_dir_size = _locked_tmp_dir_size(lock_file)
        cur_lock_file_content = tf_utils.read_file_to_string(lock_file)
        if (cur_locked_tmp_dir_size == locked_tmp_dir_size and
            cur_lock_file_content == lock_file_content):
          # There is was no data downloaded in the past
          # 'lock_file_timeout_sec'. Steal the lock and proceed with the
          # local download.
          logging.warning("Deleting lock file %s due to inactivity.",
                          lock_file)
          tf_v1.gfile.Remove(lock_file)
          break
        locked_tmp_dir_size = cur_locked_tmp_dir_size
        locked_tmp_dir_size_check_time = time.time()
        lock_file_content = cur_lock_file_content
    except tf.errors.NotFoundError:
      # Lock file or temp directory were deleted during check. Continue
      # to check whether download succeeded or we need to start our own
      # download.
      pass
    finally:
      time.sleep(5)


def atomic_download(handle,
                    download_fn,
                    module_dir,
                    lock_file_timeout_sec=10 * 60):
  """Returns the path to a Module directory for a given TF-Hub Module handle.

  Args:
    handle: (string) Location of a TF-Hub Module.
    download_fn: Callback function that actually performs download. The callback
                 receives two arguments, handle and the location of a temporary
                 directory to download the content into.
    module_dir: Directory where to download the module files to.
    lock_file_timeout_sec: The amount of time we give the current holder of
                           the lock to make progress in downloading a module.
                           If no progress is made, the lock is revoked.

  Returns:
    A string containing the path to a TF-Hub Module directory.

  Raises:
    ValueError: if the Module is not found.
    tf.errors.OpError: file I/O failures raise the appropriate subtype.
  """
  lock_file = _lock_filename(module_dir)
  task_uid = uuid.uuid4().hex
  lock_contents = _lock_file_contents(task_uid)
  tmp_dir = _temp_download_dir(module_dir, task_uid)

  # Attempt to protect against cases of processes being cancelled with
  # KeyboardInterrupt by using a try/finally clause to remove the lock
  # and tmp_dir.
  try:
    while True:
      try:
        tf_utils.atomic_write_string_to_file(lock_file, lock_contents,
                                             overwrite=False)
        # Must test condition again, since another process could have created
        # the module and deleted the old lock file since last test.
        if tf_v1.gfile.Exists(module_dir):
          # Lock file will be deleted in the finally-clause.
          return module_dir
        break  # Proceed to downloading the module.
      # These errors are believed to be permanent problems with the
      # module_dir that justify failing the download.
      except (tf.errors.NotFoundError,
              tf.errors.PermissionDeniedError,
              tf.errors.UnauthenticatedError,
              tf.errors.ResourceExhaustedError,
              tf.errors.InternalError,
              tf.errors.InvalidArgumentError,
              tf.errors.UnimplementedError):
        raise
      # All other errors are retried.
      # TODO(b/144424849): Retrying an AlreadyExistsError from the atomic write
      # should be good enough, but see discussion about misc filesystem types.
      # TODO(b/144475403): How atomic is the overwrite=False check?
      except tf.errors.OpError:
        pass

      # Wait for lock file to disappear.
      _wait_for_lock_to_disappear(handle, lock_file, lock_file_timeout_sec)
      # At this point we either deleted a lock or a lock got removed by the
      # owner or another process. Perform one more iteration of the while-loop,
      # we would either terminate due tf_v1.gfile.Exists(module_dir) or because
      # we would obtain a lock ourselves, or wait again for the lock to
      # disappear.

    # Lock file acquired.
    logging.info("Downloading TF-Hub Module '%s'.", handle)
    tf_v1.gfile.MakeDirs(tmp_dir)
    download_fn(handle, tmp_dir)
    # Write module descriptor to capture information about which module was
    # downloaded by whom and when. The file stored at the same level as a
    # directory in order to keep the content of the 'model_dir' exactly as it
    # was define by the module publisher.
    #
    # Note: The descriptor is written purely to help the end-user to identify
    # which directory belongs to which module. The descriptor is not part of the
    # module caching protocol and no code in the TF-Hub library reads its
    # content.
    _write_module_descriptor_file(handle, module_dir)
    try:
      tf_v1.gfile.Rename(tmp_dir, module_dir)
      logging.info("Downloaded TF-Hub Module '%s'.", handle)
    except tf.errors.AlreadyExistsError:
      logging.warning("Module already exists in %s", module_dir)

  finally:
    try:
      # Temp directory is owned by the current process, remove it.
      tf_v1.gfile.DeleteRecursively(tmp_dir)
    except tf.errors.NotFoundError:
      pass
    try:
      contents = tf_utils.read_file_to_string(lock_file)
    except tf.errors.NotFoundError:
      contents = ""
    if contents == lock_contents:
      # Lock file exists and is owned by this process.
      try:
        tf_v1.gfile.Remove(lock_file)
      except tf.errors.NotFoundError:
        pass

  return module_dir


class UnsupportedHandleError(Exception):
  """Exception class for incorrectly formatted handles."""


class Resolver(object):
  """Resolver base class: all resolvers inherit from this class."""
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __call__(self, handle):
    """Resolves a handle into a Module path.

    Args:
      handle: (string) the Module handle to resolve.

    Returns:
      A string representing the Module path.
    """
    pass

  @abc.abstractmethod
  def is_supported(self, handle):
    """Returns whether a handle is supported by this resolver.

    Args:
      handle: (string) the Module handle to resolve.

    Returns:
      True if the handle is properly formatted for this resolver.
      Note that a True return value does not indicate that the
      handle can be resolved, only that it is the correct format.
    """
    pass


class PathResolver(Resolver):
  """Resolves handles which are absolute paths."""

  def is_supported(self, handle):
    try:
      return tf_v1.gfile.Exists(handle)
    except tf.errors.OpError:
      return False

  def __call__(self, handle):
    return handle


class FailResolver(Resolver):
  """Always fails to resolve a path."""

  def is_supported(self, handle):
    return True

  def __call__(self, handle):
    raise UnsupportedHandleError(
        "unsupported handle format '%s'. No resolvers found that can "
        "successfully resolve it. If the handle points to the local "
        "filesystem, the error indicates that the module directory does not "
        "exist. Supported handle formats: URLs pointing to a TGZ  file "
        "(e.g. https://address/module.tgz), or Local File System directory "
        "file (e.g. /tmp/my_local_module)." % handle)
