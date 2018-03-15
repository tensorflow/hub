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
r"""Exporter tool for TF-Hub text embedding modules.

This tool creates TF-Hub Modules from embeddings text files in the following
format:
token1 1.0 2.0 3.0 4.0 5.0
token2 2.0 3.0 4.0 5.0 6.0
...

Example use:

python export.py --embedding_file=/tmp/embedding.txt --export_path=/tmp/module
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

FLAGS = None

EMBEDDINGS_VAR_NAME = "embeddings"


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
  for line in tf.gfile.GFile(file_path):
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


def make_module_spec(vocabulary_file, vocab_size, embeddings_dim,
                     num_oov_buckets, preprocess_text):
  """Makes a module spec to simply perform token to embedding lookups.

  Input of this module is a 1-D list of string tokens. For T tokens input and
  an M dimensional embedding table, the lookup result is a [T, M] shaped Tensor.

  Args:
    vocabulary_file: Text file where each line is a key in the vocabulary.
    vocab_size: The number of tokens contained in the vocabulary.
    embeddings_dim: The embedding dimension.
    num_oov_buckets: The number of out-of-vocabulary buckets.
    preprocess_text: Whether to preprocess the input tensor by removing
      punctuation and splitting on spaces.

  Returns:
    A module spec object used for constructing a TF-Hub module.
  """

  def module_fn():
    """Spec function for a token embedding module."""
    tokens = tf.placeholder(shape=[None], dtype=tf.string, name="tokens")

    embeddings_var = tf.get_variable(
        initializer=tf.zeros([vocab_size + num_oov_buckets, embeddings_dim]),
        name=EMBEDDINGS_VAR_NAME,
        dtype=tf.float32)

    lookup_table = tf.contrib.lookup.index_table_from_file(
        vocabulary_file=vocabulary_file,
        num_oov_buckets=num_oov_buckets,
    )
    ids = lookup_table.lookup(tokens)
    combined_embedding = tf.nn.embedding_lookup(params=embeddings_var, ids=ids)
    hub.add_signature("default", {"tokens": tokens},
                      {"default": combined_embedding})

  def module_fn_with_preprocessing():
    """Spec function for a full-text embedding module with preprocessing."""
    sentences = tf.placeholder(shape=[None], dtype=tf.string, name="sentences")
    # Perform a minimalistic text preprocessing by removing punctuation and
    # splitting on spaces.
    normalized_sentences = tf.regex_replace(
        input=sentences, pattern=r"\pP", rewrite="")
    tokens = tf.string_split(normalized_sentences, " ")

    # In case some of the input sentences are empty before or after
    # normalization, we will end up with empty rows. We do however want to
    # return embedding for every row, so we have to fill in the empty rows with
    # a default.
    tokens, _ = tf.sparse_fill_empty_rows(tokens, "")
    # In case all of the input sentences are empty before or after
    # normalization, we will end up with a SparseTensor with shape [?, 0]. After
    # filling in the empty rows we must ensure the shape is set properly to
    # [?, 1].
    tokens = tf.sparse_reset_shape(tokens)

    embeddings_var = tf.get_variable(
        initializer=tf.zeros([vocab_size + num_oov_buckets, embeddings_dim]),
        name=EMBEDDINGS_VAR_NAME,
        dtype=tf.float32)
    lookup_table = tf.contrib.lookup.index_table_from_file(
        vocabulary_file=vocabulary_file,
        num_oov_buckets=num_oov_buckets,
    )
    sparse_ids = tf.SparseTensor(
        indices=tokens.indices,
        values=lookup_table.lookup(tokens.values),
        dense_shape=tokens.dense_shape)

    combined_embedding = tf.nn.embedding_lookup_sparse(
        params=embeddings_var,
        sp_ids=sparse_ids,
        sp_weights=None,
        combiner="sqrtn")

    hub.add_signature("default", {"sentences": sentences},
                      {"default": combined_embedding})

  if preprocess_text:
    return hub.create_module_spec(module_fn_with_preprocessing)
  else:
    return hub.create_module_spec(module_fn)


def export(export_path, vocabulary, embeddings, num_oov_buckets,
           preprocess_text):
  """Exports a TF-Hub module that performs embedding lookups.

  Args:
    export_path: Location to export the module.
    vocabulary: List of the N tokens in the vocabulary.
    embeddings: Numpy array of shape [N+K,M] the first N rows are the
      M dimensional embeddings for the respective tokens and the next K
      rows are for the K out-of-vocabulary buckets.
    num_oov_buckets: How many out-of-vocabulary buckets to add.
    preprocess_text: Whether to preprocess the input tensor by removing
      punctuation and splitting on spaces.
  """
  # Write temporary vocab file for module construction.
  tmpdir = tempfile.mkdtemp()
  vocabulary_file = os.path.join(tmpdir, "tokens.txt")
  with tf.gfile.GFile(vocabulary_file, "w") as f:
    f.write("\n".join(vocabulary))
  vocab_size = len(vocabulary)
  embeddings_dim = embeddings.shape[1]
  spec = make_module_spec(vocabulary_file, vocab_size, embeddings_dim,
                          num_oov_buckets, preprocess_text)

  try:
    with tf.Graph().as_default():
      m = hub.Module(spec)
      # The embeddings may be very large (e.g., larger than the 2GB serialized
      # Tensor limit).  To avoid having them frozen as constant Tensors in the
      # graph we instead assign them through the placeholders and feed_dict
      # mechanism.
      p_embeddings = tf.placeholder(tf.float32)
      load_embeddings = tf.assign(m.variable_map[EMBEDDINGS_VAR_NAME],
                                  p_embeddings)

      with tf.Session() as sess:
        sess.run([load_embeddings], feed_dict={p_embeddings: embeddings})
        m.export(export_path, sess)
  finally:
    shutil.rmtree(tmpdir)


def maybe_append_oov_vectors(embeddings, num_oov_buckets):
  """Adds zero vectors for oov buckets if num_oov_buckets > 0.

  Since we are assigning zero vectors, adding more that one oov bucket is only
  meaningful if we perform fine-tuning.

  Args:
    embeddings: Embeddings to extend.
    num_oov_buckets: Number of OOV buckets in the extended embedding.
  """
  num_embeddings = np.shape(embeddings)[0]
  embedding_dim = np.shape(embeddings)[1]
  embeddings.resize(
      [num_embeddings + num_oov_buckets, embedding_dim], refcheck=False)


def export_module_from_file(embedding_file, export_path, parse_line_fn,
                            num_oov_buckets, preprocess_text):
  # Load pretrained embeddings into memory.
  vocabulary, embeddings = load(embedding_file, parse_line_fn)

  # Add OOV buckets if num_oov_buckets > 0.
  maybe_append_oov_vectors(embeddings, num_oov_buckets)

  # Export the embedding vectors into a TF-Hub module.
  export(export_path, vocabulary, embeddings, num_oov_buckets, preprocess_text)


def main(_):
  export_module_from_file(FLAGS.embedding_file, FLAGS.export_path, parse_line,
                          FLAGS.num_oov_buckets, FLAGS.preprocess_text)


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
      "--preprocess_text",
      type=bool,
      default=False,
      help="Whether to preprocess the input tensor by removing punctuation and "
      "splitting on spaces. Use this if input is a dense tensor of untokenized "
      "sentences.")
  parser.add_argument(
      "--num_oov_buckets",
      type=int,
      default="1",
      help="How many OOV buckets to add.")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
