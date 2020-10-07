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

import hashlib
import urllib

import tensorflow as tf
from tensorflow_hub import resolver


LOCK_FILE_TIMEOUT_SEC = 10 * 60  # 10 minutes

_COMPRESSED_FORMAT_QUERY = ("tf-hub-format", "compressed")


def _module_dir(handle):
  """Returns the directory where to cache the module."""
  cache_dir = resolver.tfhub_cache_dir(use_temp=True)
  return resolver.create_local_module_dir(
      cache_dir,
      hashlib.sha1(handle.encode("utf8")).hexdigest())


def _is_tarfile(filename):
  """Returns true if 'filename' is TAR file."""
  return filename.endswith((".tar", ".tar.gz", ".tgz"))


class HttpCompressedFileResolver(resolver.HttpResolverBase):
  """Resolves HTTP handles by downloading and decompressing them to local fs."""

  def is_supported(self, handle):
    # HTTP(S) handles are assumed to point to tarfiles.
    return self.is_http_protocol(handle)

  def __call__(self, handle):
    module_dir = _module_dir(handle)

    def download(handle, tmp_dir):
      """Fetch a module via HTTP(S), handling redirect and download headers."""
      request = urllib.request.Request(
          self._append_compressed_format_query(handle))
      response = self._call_urlopen(request)
      return resolver.DownloadManager(handle).download_and_uncompress(
          response, tmp_dir)

    return resolver.atomic_download(handle, download, module_dir,
                                    self._lock_file_timeout_sec())

  def _lock_file_timeout_sec(self):
    # This method is provided as a convenience to simplify testing.
    return LOCK_FILE_TIMEOUT_SEC

  def _append_compressed_format_query(self, handle):
    return self._append_format_query(handle, _COMPRESSED_FORMAT_QUERY)


class GcsCompressedFileResolver(resolver.Resolver):
  """Resolves GCS handles by downloading and decompressing them to local fs."""

  def is_supported(self, handle):
    return handle.startswith("gs://") and _is_tarfile(handle)

  def __call__(self, handle):
    module_dir = _module_dir(handle)

    def download(handle, tmp_dir):
      return resolver.DownloadManager(handle).download_and_uncompress(
          tf.compat.v1.gfile.GFile(handle, "rb"), tmp_dir)

    return resolver.atomic_download(handle, download, module_dir,
                                    LOCK_FILE_TIMEOUT_SEC)
