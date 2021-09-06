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
"""Tests for tensorflow_hub.uncompressed_module_resolver."""

import io
from unittest import mock
import urllib

import tensorflow as tf
from tensorflow_hub import resolver
from tensorflow_hub import test_utils
from tensorflow_hub import uncompressed_module_resolver


class UncompressedModuleResolverTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.handles = ["http://example.com/module", "https://example.com/module"]
    # pylint: disable=line-too-long
    self.uncompressed_resolver = uncompressed_module_resolver.HttpUncompressedFileResolver(
    )

  def test_append_format_query(self):
    tests = [
        (
            "https://example.com/module",
            "https://example.com/module?tf-hub-format=uncompressed",
        ),
        (
            "https://example.com/module?extra=abc",
            "https://example.com/module?extra=abc&tf-hub-format=uncompressed",
        ),
        (
            "https://example.com/module?extra=abc",
            "https://example.com/module?extra=abc&tf-hub-format=uncompressed",
        ),
        (
            "https://example.com/module?extra=abc&tf-hub-format=test",
            ("https://example.com/module?extra=abc&"
             "tf-hub-format=test&tf-hub-format=uncompressed"),
        )
    ]
    for handle, expected in tests:
      self.assertTrue(
          self.uncompressed_resolver._append_uncompressed_format_query(handle),
          expected)

  def test_wrong_protocol(self):
    handles = [
        "foo.tar", "gs://foo.tar", "gs://model/", "gs://model/uncompressed.tgz"
    ]
    for handle in handles:
      self.assertFalse(self.uncompressed_resolver.is_supported(handle))

  def test_on_compressed_load_format(self):
    with test_utils.CompressedLoadFormatContext():
      for handle in self.handles:
        self.assertFalse(self.uncompressed_resolver.is_supported(handle))

  def test_on_uncompressed_load_format(self):
    with test_utils.UncompressedLoadFormatContext():
      for handle in self.handles:
        self.assertTrue(self.uncompressed_resolver.is_supported(handle))

  def test_on_auto_load_format_default(self):
    with test_utils.AutoLoadFormatContext():
      for handle in self.handles:
        self.assertFalse(self.uncompressed_resolver.is_supported(handle))

  def test_server_returns_303_but_no_gcs_path(self):
    http_error = urllib.error.HTTPError(None, 303, None, None,
                                        io.BytesIO(b"file://somefile"))
    with mock.patch.object(
        resolver.HttpResolverBase, "_call_urlopen", side_effect=http_error):
      with self.assertRaisesWithLiteralMatch(
          ValueError, "Expected server to return a GCS location but received "
          "file://somefile"):
        self.uncompressed_resolver("https://tfhub.dev/google/model/1")

  def test_server_returns_200(self):
    mockresponse = mock.Mock()
    mockresponse.code = 200
    with mock.patch.object(
        resolver.HttpResolverBase, "_call_urlopen", return_value=mockresponse):
      with self.assertRaisesWithLiteralMatch(
          ValueError,
          "Expected 303 See other HTTP response but received code 200"):
        self.uncompressed_resolver("https://tfhub.dev/google/model/1")

  def test_server_returns_unexpected_error(self):
    http_error = urllib.error.HTTPError(None, 404, None, None, None)
    with mock.patch.object(
        resolver.HttpResolverBase, "_call_urlopen", side_effect=http_error):
      with self.assertRaisesWithLiteralMatch(
          ValueError,
          "Expected 303 See other HTTP response but received code 404"):
        self.uncompressed_resolver("https://tfhub.dev/google/model/1")


if __name__ == "__main__":
  tf.test.main()
