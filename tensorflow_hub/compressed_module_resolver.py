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
"""Functions to resolve TF-Hub Module stored in compressed TGZ format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
# pylint:disable=g-import-not-at-top
try:
  import urllib.request as url
except ImportError:
  import urllib2 as url
# pylint:disable=g-import-not-at-top

import tensorflow as tf

from tensorflow_hub import resolver


LOCK_FILE_TIMEOUT_SEC = 10 * 60  # 10 minutes


def _module_dir(cache_dir, handle):
  """Returns the directory where to cache the module."""
  return resolver.create_local_module_dir(
      cache_dir,
      hashlib.sha1(handle.encode("utf8")).hexdigest())


def _is_tarfile(filename):
  """Returns true if 'filename' is TAR file."""
  return (filename.endswith(".tar") or filename.endswith(".tar.gz") or
          filename.endswith(".tgz"))


class HttpCompressedFileResolver(resolver.Resolver):
  """Resolves HTTP handles by downloading and decompressing them to local fs."""

  def __init__(self, cache_dir=None):
    """Creates a resolver that streams tar/gz file content over HTTP.

    Args:
      cache_dir: directory to download and cache modules to.
    """
    self._cache_dir = resolver.tfhub_cache_dir(cache_dir, use_temp=True)

  def is_supported(self, handle):
    return ((handle.startswith("http://") or handle.startswith("https://")) and
            _is_tarfile(handle))

  def _get_module_path(self, handle):
    module_dir = _module_dir(self._cache_dir, handle)

    def download(handle, tmp_dir):
      request = url.Request(handle)
      url_opener = url.build_opener(url.HTTPRedirectHandler)
      return resolver.download_and_uncompress(
          url_opener.open(request), tmp_dir)

    return resolver.atomic_download(handle, download, module_dir,
                                    self._lock_file_timeout_sec())

  def _lock_file_timeout_sec(self):
    # This method is provided as a convenience to simplify testing.
    return LOCK_FILE_TIMEOUT_SEC


class GcsCompressedFileResolver(resolver.Resolver):
  """Resolves GCS handles by downloading and decompressing them to local fs."""

  def __init__(self, cache_dir=None):
    """Creates a resolver that streams tar/gz file content from GCS.

    Args:
      cache_dir: directory to download and cache modules to.
    """
    self._cache_dir = resolver.tfhub_cache_dir(cache_dir, use_temp=True)

  def is_supported(self, handle):
    return handle.startswith("gs://") and _is_tarfile(handle)

  def _get_module_path(self, handle):
    module_dir = _module_dir(self._cache_dir, handle)

    def download(handle, tmp_dir):
      return resolver.download_and_uncompress(
          tf.gfile.GFile(handle, "r"), tmp_dir)

    return resolver.atomic_download(handle, download, module_dir,
                                    LOCK_FILE_TIMEOUT_SEC)


def get_default():
  """Returns the default compressed module-based handle resolver for TF-Hub."""
  return resolver.UseFirstSupportingResolver(
      resolvers=[
          HttpCompressedFileResolver(),
          GcsCompressedFileResolver(),
          resolver.PathResolver()])
