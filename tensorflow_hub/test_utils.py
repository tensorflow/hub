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

import os
import socket
import sys
import threading

from absl import flags
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_hub import resolver


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

    class TCPServerV6(socketserver.TCPServer):

      address_family = socket.AF_INET6

    class RequestHandler(http.server.SimpleHTTPRequestHandler):

      def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed_url.query)
        if qs["tf-hub-format"][0] == "compressed":
          _do_redirect(self, download_url)
        else:
          _do_documentation(self)

    server = TCPServerV6(("", 0), RequestHandler)
    _, server_port, _, _ = server.server_address
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

    class TCPServerV6(socketserver.TCPServer):

      address_family = socket.AF_INET6

    class RedirectHandler(http.server.SimpleHTTPRequestHandler):

      def do_GET(self):
        _do_redirect(self, redirect)

    server = TCPServerV6(("", 0), RedirectHandler if redirect else
                         http.server.SimpleHTTPRequestHandler)
    _, server_port, _, _ = server.server_address
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


def get_test_data_path(file_or_dirname):
  """Return full test data path."""
  for directory, subdirs, files in tf.io.gfile.walk(test_srcdir()):
    for f in subdirs + files:
      if f.endswith(file_or_dirname):
        return os.path.join(directory, f)
  raise ValueError("No %s in test directory" % file_or_dirname)


def export_module(module_export_path):
  """Create and export a simple module to the specified path."""

  def _stateless_module_fn():
    """Simple module that squares an input."""
    x = tf.compat.v1.placeholder(tf.int64)
    y = x * x
    hub.add_signature(inputs=x, outputs=y)

  spec = hub.create_module_spec(_stateless_module_fn)
  m = hub.Module(spec, name="test_module")
  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    m.export(module_export_path, sess)


class EnvVariableContextManager(object):
  """Set an environment variable for the context and unset it afterwards."""

  def __init__(self, key, value):
    self.key = key
    self.value = value

  def __enter__(self):
    os.environ[self.key] = self.value
    return self

  def __exit__(self, exc_type, exc_value, exc_traceback):
    del os.environ[self.key]
    return True


class CompressedLoadFormatContext(EnvVariableContextManager):
  """Set the load format to COMPRESSED during the execution of the context."""

  def __init__(self):
    super().__init__(resolver._TFHUB_MODEL_LOAD_FORMAT,
                     resolver.ModelLoadFormat.COMPRESSED.value)


class UncompressedLoadFormatContext(EnvVariableContextManager):
  """Set the load format to UNCOMPRESSED during the execution of the context."""

  def __init__(self):
    super().__init__(resolver._TFHUB_MODEL_LOAD_FORMAT,
                     resolver.ModelLoadFormat.UNCOMPRESSED.value)


class AutoLoadFormatContext(EnvVariableContextManager):
  """Set the load format to AUTO during the execution of the context."""

  def __init__(self):
    super().__init__(resolver._TFHUB_MODEL_LOAD_FORMAT,
                     resolver.ModelLoadFormat.AUTO.value)
