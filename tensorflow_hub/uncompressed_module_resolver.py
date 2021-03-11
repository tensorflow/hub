# Copyright 2020 The TensorFlow Hub Authors. All Rights Reserved.
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
"""Functions to resolve TF-Hub Modules stored in uncompressed folders on GCS."""
import urllib

from tensorflow_hub import resolver

_UNCOMPRESSED_FORMAT_QUERY = ("tf-hub-format", "uncompressed")


class HttpUncompressedFileResolver(resolver.HttpResolverBase):
  """Resolves HTTP handles by requesting and reading their GCS location."""

  def __init__(self):
    super().__init__()
    self.path_resolver = resolver.PathResolver()

  def __call__(self, handle):
    """Request the gs:// path for the handle and pass it to PathResolver."""
    handle_with_params = self._append_uncompressed_format_query(handle)
    gcs_location = self._request_gcs_location(handle_with_params)
    return self.path_resolver(gcs_location)

  def _append_uncompressed_format_query(self, handle):
    return self._append_format_query(handle, _UNCOMPRESSED_FORMAT_QUERY)

  def _request_gcs_location(self, handle_with_params):
    """Request ...?tf-hub-format=uncompressed and return the response body."""
    request = urllib.request.Request(handle_with_params)
    gcs_location = self._call_urlopen(request)
    if not gcs_location.startswith("gs://"):
      raise ValueError(
          "Expected server to return a GCS location but received {}".format(
              gcs_location))
    return gcs_location

  def _call_urlopen(self, request):
    """We expect a '303 See other' response.

    Fail on anything else.

    Args:
      request: Request to the ...?tf-hub-format=uncompressed URL.

    Returns:
      String containing the server response

    Raise a ValueError if
    - a HTTPError != 303 occurrs
    - urlopen does not raise an HTTPError (on 2xx responses)
    """

    def raise_on_unexpected_code(code):
      raise ValueError(
          "Expected 303 See other HTTP response but received code {}".format(
              code))

    try:
      response = super()._call_urlopen(request)
      raise_on_unexpected_code(response.code)
    except urllib.error.HTTPError as error:
      if error.code != 303:
        raise_on_unexpected_code(error.code)
      return error.read().decode()

  def is_supported(self, handle):
    if not self.is_http_protocol(handle):
      return False
    load_format = resolver.model_load_format()
    return load_format == resolver.ModelLoadFormat.UNCOMPRESSED.value
