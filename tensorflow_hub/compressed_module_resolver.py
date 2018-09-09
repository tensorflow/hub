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
  import urllib.parse as urlparse
  from urllib.parse import urlencode
except ImportError:
  import urllib2 as url
  from urllib import urlencode
  import urlparse
# pylint:disable=g-import-not-at-top

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
  return (filename.endswith(".tar") or filename.endswith(".tar.gz") or
          filename.endswith(".tgz"))


def _append_compressed_format_query(handle):
  # Convert the tuple from urlparse into list so it can be updated in place.
  parsed = list(urlparse.urlparse(handle))
  qsl = urlparse.parse_qsl(parsed[4])
  qsl.append(_COMPRESSED_FORMAT_QUERY)
  # NOTE: Cast to string to avoid urlunparse to deal with mixed types.
  # This happens due to backport of urllib.parse into python2 returning an
  # instance of <class 'future.types.newstr.newstr'>.
  parsed[4] = str(urlencode(qsl))
  return urlparse.urlunparse(parsed)


class HttpCompressedFileResolver(resolver.Resolver):
  """Resolves HTTP handles by downloading and decompressing them to local fs."""

  def is_supported(self, handle):
    # HTTP(S) handles are assumed to point to tarfiles.
    return handle.startswith("http://") or handle.startswith("https://")

  def __call__(self, handle):
    module_dir = _module_dir(handle)

    def download(handle, tmp_dir):
      """Fetch a module via HTTP(S), handling redirect and download headers."""
      cur_url = handle
      request = url.Request(_append_compressed_format_query(handle))

      # Look for and handle a special response header. If present, interpret it
      # as a redirect to the module download location. This allows publishers
      # (if they choose) to provide the same URL for both a module download and
      # its documentation.

      class LoggingHTTPRedirectHandler(url.HTTPRedirectHandler):

        def redirect_request(self, req, fp, code, msg, headers, newurl):
          cur_url = newurl  # pylint:disable=unused-variable
          return url.HTTPRedirectHandler.redirect_request(
              self, req, fp, code, msg, headers, newurl)

      url_opener = url.build_opener(LoggingHTTPRedirectHandler)
      response = url_opener.open(request)
      return resolver.DownloadManager(cur_url).download_and_uncompress(
          response, tmp_dir)

    return resolver.atomic_download(handle, download, module_dir,
                                    self._lock_file_timeout_sec())

  def _lock_file_timeout_sec(self):
    # This method is provided as a convenience to simplify testing.
    return LOCK_FILE_TIMEOUT_SEC


class GcsCompressedFileResolver(resolver.Resolver):
  """Resolves GCS handles by downloading and decompressing them to local fs."""

  def is_supported(self, handle):
    return handle.startswith("gs://") and _is_tarfile(handle)

  def __call__(self, handle):
    module_dir = _module_dir(handle)

    def download(handle, tmp_dir):
      return resolver.DownloadManager(handle).download_and_uncompress(
          tf.gfile.GFile(handle, "r"), tmp_dir)

    return resolver.atomic_download(handle, download, module_dir,
                                    LOCK_FILE_TIMEOUT_SEC)
