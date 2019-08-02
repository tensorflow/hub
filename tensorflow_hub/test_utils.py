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
"""Common testing functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import socket
import sys
import threading

from absl import flags

# TODO(b/73987364): It is not possible to extend feature columns without
# depending on TensorFlow internal implementation details.
# pylint: disable=g-direct-tensorflow-import
# pylint: disable=g-import-not-at-top,g-statement-before-imports
try:
  from tensorflow.python.feature_column import dense_features_v2
except ImportError:
  pass
# pylint: disable=g-import-not-at-top,g-statement-before-imports
from tensorflow.python.feature_column import feature_column_v2
# pylint: enable=g-direct-tensorflow-import


def _do_redirect(handler, location):
  handler.send_response(301)
  handler.send_header("Location", location)
  handler.end_headers()


def _do_documentation(handler):
  handler.send_response(200)
  handler.end_headers()
  handler.wfile.write(b"Here is some documentation.")


def start_smart_module_server(download_url):
  """Serve documentation and module requests at the same URL."""
  # pylint:disable=g-import-not-at-top
  if sys.version_info[0] == 2:
    import BaseHTTPServer
    import SimpleHTTPServer
    import urlparse

    class HTTPServerV6(BaseHTTPServer.HTTPServer):

      address_family = socket.AF_INET6

    class RequestHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):

      def do_GET(self):
        parsed_url = urlparse.urlparse(self.path)
        qs = urlparse.parse_qs(parsed_url.query)
        if qs["tf-hub-format"][0] == "compressed":
          _do_redirect(self, download_url)
        else:
          _do_documentation(self)

    server = HTTPServerV6(("", 0), RequestHandler)
    server_port = server.server_port
  else:
    import http.server
    import socketserver
    import urllib

    class RequestHandler(http.server.SimpleHTTPRequestHandler):

      def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed_url.query)
        if qs["tf-hub-format"][0] == "compressed":
          _do_redirect(self, download_url)
        else:
          _do_documentation(self)

    server = socketserver.TCPServer(("", 0), RequestHandler)
    _, server_port = server.server_address
  # pylint:disable=g-import-not-at-top

  thread = threading.Thread(target=server.serve_forever)
  thread.daemon = True
  thread.start()

  return server_port


def start_http_server(redirect=None):
  """Returns the port of the newly started HTTP server."""

  # Start HTTP server to serve TAR files.
  # pylint:disable=g-import-not-at-top
  if sys.version_info[0] == 2:
    import BaseHTTPServer
    import SimpleHTTPServer

    class HTTPServerV6(BaseHTTPServer.HTTPServer):

      address_family = socket.AF_INET6

    class RedirectHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):

      def do_GET(self):
        _do_redirect(self, redirect)

    server = HTTPServerV6(("", 0), RedirectHandler if redirect else
                          SimpleHTTPServer.SimpleHTTPRequestHandler)
    server_port = server.server_port
  else:
    import http.server
    import socketserver

    class RedirectHandler(http.server.SimpleHTTPRequestHandler):

      def do_GET(self):
        _do_redirect(self, redirect)

    server = socketserver.TCPServer(("", 0), RedirectHandler if redirect else
                                    http.server.SimpleHTTPRequestHandler)
    _, server_port = server.server_address
  # pylint:disable=g-import-not-at-top

  thread = threading.Thread(target=server.serve_forever)
  thread.daemon = True
  thread.start()

  return server_port


def test_srcdir():
  """Returns the path where to look for test data files."""
  if "test_srcdir" in flags.FLAGS:
    return flags.FLAGS["test_srcdir"].value
  elif "TEST_SRCDIR" in os.environ:
    return os.environ["TEST_SRCDIR"]
  else:
    raise RuntimeError("Missing TEST_SRCDIR environment.")


def get_dense_features_module():
  """Returns the module that contains DenseFeatures class.

  This is a defense against changes in https://github.com/tensorflow/tensorflow/commit/64586f18724f737393071125a91b19adf013cf8a  # pylint: disable=line-too-long
  that moved DenseFeatures into dense_features_v2.
  """
  if hasattr(feature_column_v2, "DenseFeatures"):
    return feature_column_v2
  return dense_features_v2
