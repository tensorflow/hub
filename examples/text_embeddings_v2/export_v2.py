# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Exporter tool for TF-Hub text embedding modules.

This tool creates TF-Hub modules (equivalent to TensorFlow 2.X SavedModel) from embeddings text files in the following
format:
token1 1.0 2.0 3.0 4.0 5.0
token2 2.0 3.0 4.0 5.0 6.0
...

Example use:

python export.py --embedding_file=/tmp/embedding.txt --export_path=/tmp/module

This currently depends on TF 2.0.0-beta0.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import tempfile
import argparse
from absl import app
from absl import flags

import tensorflow as tf

# Importing symbols that are not public as of now.
from tensorflow.python.ops import lookup_ops
from tensorflow.python.training.tracking import tracking

FLAGS = None

def parse_line(line):
  """Parses a line of a text embedding file.

  Args:
    line: (str) One line of the text embedding file.

  Returns:
    A token string and its embedding vector in floats.
  """
  columns = line.split()
  token = columns.pop(0)
  values = [float(column) for column in columns]
  return token, values


def load(file_path, parse_line_fn):
  """Loads a text embedding into memory as a numpy matrix.

  Args:
    file_path: Path to the text embedding file.
    parse_line_fn: callback function to parse each file line.

  Returns:
    A tuple of (list of vocabulary tokens, numpy matrix of embedding vectors).

  Raises:
    ValueError: if the data in the sstable is inconsistent.
  """
  vocabulary = []
  embeddings = []
  embeddings_dim = None
  for line in tf.io.gfile.GFile(file_path):
    token, embedding = parse_line_fn(line)
    if not embeddings_dim:
      embeddings_dim = len(embedding)
    elif embeddings_dim != len(embedding):
      raise ValueError(
          "Inconsistent embedding dimension detected, %d != %d for token %s",
          embeddings_dim, len(embedding), token)

    vocabulary.append(token)
    embeddings.append(embedding)

  return vocabulary, np.array(embeddings)


def write_vocabulary_file(vocabulary):
  """Write temporary vocab file for module construction."""
  tmpdir = tempfile.mkdtemp()
  vocabulary_file = os.path.join(tmpdir, "tokens.txt")
  with tf.io.gfile.GFile(vocabulary_file, "w") as f:
    for entry in vocabulary:
      f.write(entry + "\n")
  return vocabulary_file


class TextEmbeddingModel(tf.train.Checkpoint):
  """Text embedding model.

  A text embeddings model that takes a sentences on input and outputs the
  sentence embedding.
  """

  def __init__(self, vocab_file_path, oov_buckets):
    super(TextEmbeddingModel, self).__init__()
    self._vocabulary, self._pretrained_vectors = load(vocab_file_path,parse_line)
    self._oov_buckets = oov_buckets
    # Make the vocabulary file a `TrackableAsset` to ensure it is saved along with the model.
    self._vocabulary_file = tracking.TrackableAsset(
        write_vocabulary_file(self._vocabulary))
    self._table = lookup_ops.index_table_from_file(
        vocabulary_file=self._vocabulary_file,
        num_oov_buckets=self._oov_buckets,
        hasher_spec=lookup_ops.FastHashSpec)
    oovs = np.zeros([oov_buckets, self._pretrained_vectors.shape[1]])
    self._pretrained_vectors.resize(
	[self._pretrained_vectors.shape[0] + oov_buckets, 
	self._pretrained_vectors.shape[1]])
    self._pretrained_vectors[self._pretrained_vectors.shape[0]:,:] = oovs
    self.embeddings = tf.Variable(self._pretrained_vectors)
    self.variables = [self.embeddings]
    self.trainable_variables = self.variables
  
  @tf.function(input_signature=[tf.TensorSpec([None], tf.dtypes.string)])
  def _tokenize(self, sentences):
    # Perform a minimalistic text preprocessing by removing punctuation and
    # splitting on spaces.
    normalized_sentences = tf.strings.regex_replace(
        input=sentences, pattern=r"\pP", rewrite="")
    normalized_sentences = tf.reshape(normalized_sentences, [-1])
    sparse_tokens = tf.strings.split(normalized_sentences, " ").to_sparse()

    # Deal with a corner case: there is one empty sentence.
    sparse_tokens, _ = tf.sparse.fill_empty_rows(sparse_tokens, tf.constant(""))
    # Deal with a corner case: all sentences are empty.
    sparse_tokens = tf.sparse.reset_shape(sparse_tokens)
    sparse_token_ids = self._table.lookup(sparse_tokens.values)

    return (sparse_tokens.indices, sparse_token_ids, sparse_tokens.dense_shape)

  @tf.function(input_signature=[tf.TensorSpec([None], tf.dtypes.string)])
  def __call__(self, sentences):
    
    token_ids, token_values, token_dense_shape = self._tokenize(sentences)

    return tf.nn.safe_embedding_lookup_sparse(
        embedding_weights=self.embeddings,
        sparse_ids=tf.SparseTensor(token_ids, token_values, token_dense_shape),
        sparse_weights=None,
        combiner="sqrtn")

def export_module_from_file(embedding_file,num_oov_buckets,export_path):
    module = TextEmbeddingModel(embedding_file,num_oov_buckets)
    tf.saved_model.save(module, export_path)

def main(_):
    export_module_from_file(FLAGS.embedding_file, FLAGS.num_oov_buckets, FLAGS.export_path)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--embedding_file",
      type=str,
      default=None,
      help="Path to file with embeddings.")
  parser.add_argument(
      "--export_path",
      type=str,
      default=None,
      help="Where to export the module.")
  parser.add_argument(
      "--num_oov_buckets",
      type=int,
      default="1",
      help="How many OOV buckets to add.")
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)