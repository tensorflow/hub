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
"""End-to-end tests for tensorflow_hub.module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow_hub import resolver
from tensorflow_hub import test_utils


class ModuleEnd2EndTest(tf.test.TestCase):

  def setUp(self):
    # Set current directory to test temp directory where we can create
    # files and serve them through the HTTP server.
    os.chdir(self.get_temp_dir())

    self.server_port = test_utils.start_http_server()

  def _stateless_module_fn(self):
    """Simple module that squares an input."""
    x = tf.placeholder(tf.int64)
    y = x*x
    hub.add_signature(inputs=x, outputs=y)

  def _list_module_files(self, module_dir):
    files = []
    for f in tf.gfile.ListDirectory(module_dir):
      full_path = os.path.join(module_dir, f)
      stat_res = tf.gfile.Stat(full_path)
      if stat_res.is_directory:
        files.extend(self._list_module_files(full_path))
      else:
        files.append(f)
    return files

  def testHttpLocations(self):
    spec = hub.create_module_spec(self._stateless_module_fn)
    m = hub.Module(spec, name="test_module")
    out = m(10)

    export_path = os.path.join(self.get_temp_dir(), "module")
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllClose(sess.run(out), 100)
      m.export(export_path, sess)

    os.chdir(export_path)

    tar = tarfile.open("test_module.tgz", "w")
    for f in self._list_module_files(export_path):
      tar.add(f)
    tar.close()

    m = hub.Module("http://localhost:%d/test_module.tgz" % self.server_port)
    out = m(11)
    with tf.Session() as sess:
      self.assertAllClose(sess.run(out), 121)

  def testUnknownHandleFormat(self):
    try:
      hub.Module("s3://my_module.zip")
    except resolver.UnsupportedHandleError as e:
      self.assertStartsWith(
          str(e), "unsupported handle format 's3://my_module.zip'. No "
          "resolvers found that can successfully resolve it.")
      self.assertNotEquals(-1, str(e).find("Currently supported handle"))

    try:
      non_existant_module = os.path.join(self.get_temp_dir(), "missing_module")
      hub.Module(non_existant_module)
    except resolver.UnsupportedHandleError as e:
      self.assertStartsWith(
          str(e), "unsupported handle format '%s'. No "
          "resolvers found that can successfully resolve it." %
          non_existant_module)
      self.assertNotEquals(-1, str(e).find("Currently supported handle"))


if __name__ == "__main__":
  tf.test.main()
