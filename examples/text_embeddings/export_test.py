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
"""Tests for text embedding exporting tool."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from examples.text_embeddings import export

_MOCK_EMBEDDING = "\n".join(
    ["cat 1.11 2.56 3.45", "dog 1 2 3", "mouse 0.5 0.1 0.6"])


class ExportTokenEmbeddingTest(tf.test.TestCase):

  def setUp(self):
    self._embedding_file_path = os.path.join(self.get_temp_dir(),
                                             "mock_embedding_file.txt")
    with tf.gfile.GFile(self._embedding_file_path, mode="w") as f:
      f.write(_MOCK_EMBEDDING)

  def testEmbeddingLoaded(self):
    vocabulary, embeddings = export.load(self._embedding_file_path,
                                         export.parse_line)
    self.assertEqual((3,), np.shape(vocabulary))
    self.assertEqual((3, 3), np.shape(embeddings))

  def testExportTokenEmbeddingModule(self):
    export.export_module_from_file(
        embedding_file=self._embedding_file_path,
        export_path=self.get_temp_dir(),
        parse_line_fn=export.parse_line,
        num_oov_buckets=1,
        preprocess_text=False)
    with tf.Graph().as_default():
      hub_module = hub.Module(self.get_temp_dir())
      tokens = tf.constant(["cat", "lizard", "dog"])
      embeddings = hub_module(tokens)
      with tf.Session() as session:
        session.run(tf.tables_initializer())
        session.run(tf.global_variables_initializer())
        self.assertAllClose(
            session.run(embeddings),
            [[1.11, 2.56, 3.45], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])

  def testExportFulltextEmbeddingModule(self):
    export.export_module_from_file(
        embedding_file=self._embedding_file_path,
        export_path=self.get_temp_dir(),
        parse_line_fn=export.parse_line,
        num_oov_buckets=1,
        preprocess_text=True)
    with tf.Graph().as_default():
      hub_module = hub.Module(self.get_temp_dir())
      tokens = tf.constant(["cat", "cat cat", "lizard. dog", "cat? dog", ""])
      embeddings = hub_module(tokens)
      with tf.Session() as session:
        session.run(tf.tables_initializer())
        session.run(tf.global_variables_initializer())
        self.assertAllClose(
            session.run(embeddings),
            [[1.11, 2.56, 3.45], [1.57, 3.62, 4.88], [0.70, 1.41, 2.12],
             [1.49, 3.22, 4.56], [0.0, 0.0, 0.0]],
            rtol=0.02)


if __name__ == "__main__":
  tf.test.main()
