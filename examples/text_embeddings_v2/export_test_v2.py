# Copyright 2019 The TensorFlow Hub Authors. All Rights Reserved.
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
"""Tests for text embedding exporting tool v2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from examples.text_embeddings_v2 import export_v2

_MOCK_EMBEDDING = "\n".join(
    ["cat 1.11 2.56 3.45", "dog 1 2 3", "mouse 0.5 0.1 0.6"])


class ExportTokenEmbeddingTest(tf.test.TestCase):
  """Test for text embedding exporter."""

  def setUp(self):
    self._embedding_file_path = os.path.join(self.get_temp_dir(),
                                             "mock_embedding_file.txt")
    with tf.io.gfile.GFile(self._embedding_file_path, mode="w") as f:
      f.write(_MOCK_EMBEDDING)

  def testEmbeddingLoaded(self):
    vocabulary, embeddings = export_v2.load(self._embedding_file_path,
                                            export_v2.parse_line,
                                            num_lines_to_ignore=0,
                                            num_lines_to_use=None)
    self.assertEqual((3,), np.shape(vocabulary))
    self.assertEqual((3, 3), np.shape(embeddings))

  def testExportTextEmbeddingModule(self):
    export_v2.export_module_from_file(
        embedding_file=self._embedding_file_path,
        export_path=self.get_temp_dir(),
        num_oov_buckets=1,
        num_lines_to_ignore=0,
        num_lines_to_use=None)
    hub_module = hub.load(self.get_temp_dir())
    tokens = tf.constant(["cat", "cat cat", "lizard. dog", "cat? dog", ""])
    embeddings = hub_module(tokens)
    self.assertAllClose(
        embeddings.numpy(),
        [[1.11, 2.56, 3.45], [1.57, 3.62, 4.88], [0.70, 1.41, 2.12],
         [1.49, 3.22, 4.56], [0.0, 0.0, 0.0]],
        rtol=0.02)

  def testEmptyInput(self):
    export_v2.export_module_from_file(
        embedding_file=self._embedding_file_path,
        export_path=self.get_temp_dir(),
        num_oov_buckets=1,
        num_lines_to_ignore=0,
        num_lines_to_use=None)
    hub_module = hub.load(self.get_temp_dir())
    tokens = tf.constant(["", "", ""])
    embeddings = hub_module(tokens)
    self.assertAllClose(
        embeddings.numpy(), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        rtol=0.02)

  def testEmptyLeading(self):
    export_v2.export_module_from_file(
        embedding_file=self._embedding_file_path,
        export_path=self.get_temp_dir(),
        num_oov_buckets=1,
        num_lines_to_ignore=0,
        num_lines_to_use=None)
    hub_module = hub.load(self.get_temp_dir())
    tokens = tf.constant(["", "cat dog"])
    embeddings = hub_module(tokens)
    self.assertAllClose(
        embeddings.numpy(), [[0.0, 0.0, 0.0], [1.49, 3.22, 4.56]], rtol=0.02)

  def testNumLinesIgnore(self):
    export_v2.export_module_from_file(
        embedding_file=self._embedding_file_path,
        export_path=self.get_temp_dir(),
        num_oov_buckets=1,
        num_lines_to_ignore=1,
        num_lines_to_use=None)
    hub_module = hub.load(self.get_temp_dir())
    tokens = tf.constant(["cat", "dog", "mouse"])
    embeddings = hub_module(tokens)
    self.assertAllClose(
        embeddings.numpy(), [[0.0, 0.0, 0.0], [1, 2, 3], [0.5, 0.1, 0.6]],
        rtol=0.02)

  def testNumLinesUse(self):
    export_v2.export_module_from_file(
        embedding_file=self._embedding_file_path,
        export_path=self.get_temp_dir(),
        num_oov_buckets=1,
        num_lines_to_ignore=0,
        num_lines_to_use=2)
    hub_module = hub.load(self.get_temp_dir())
    tokens = tf.constant(["cat", "dog", "mouse"])
    embeddings = hub_module(tokens)
    self.assertAllClose(
        embeddings.numpy(), [[1.1, 2.56, 3.45], [1, 2, 3], [0, 0, 0]],
        rtol=0.02)


if __name__ == "__main__":
  # This test is only supported in TF 2.0+.
  if tf.executing_eagerly():
    logging.info("Using TF version: %s", tf.__version__)
    tf.test.main()
  else:
    logging.warning("Skipping running tests for TF Version: %s", tf.__version__)
