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

import tensorflow as tf


def start_http_server():
  """Returns the port of the newly started HTTP server."""

  # Start HTTP server to serve TAR files.
  # pylint:disable=g-import-not-at-top
  if sys.version_info[0] == 2:
    import BaseHTTPServer
    import SimpleHTTPServer

    class HTTPServerV6(BaseHTTPServer.HTTPServer):

      address_family = socket.AF_INET6

    server = HTTPServerV6(("", 0),
                          SimpleHTTPServer.SimpleHTTPRequestHandler)
    server_port = server.server_port
  else:
    import http.server
    import socketserver
    server = socketserver.TCPServer(
        ("", 0), http.server.SimpleHTTPRequestHandler)
    _, server_port = server.server_address
  # pylint:disable=g-import-not-at-top

  thread = threading.Thread(target=server.serve_forever)
  thread.daemon = True
  thread.start()

  return server_port


def test_srcdir():
  """Returns the path where to look for test data files."""
  if "test_srcdir" in tf.app.flags.FLAGS:
    return tf.app.flags.FLAGS["test_srcdir"].value
  elif "TEST_SRCDIR" in os.environ:
    return os.environ["TEST_SRCDIR"]
  else:
    raise RuntimeError("Missing TEST_SRCDIR environment.")
