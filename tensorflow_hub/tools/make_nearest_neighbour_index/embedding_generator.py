# Copyright 2019 The TensorFlow Hub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generates embedding using a TF-Hub module."""

import os
import pickle

import apache_beam as beam
from apache_beam.transforms import util

# TODO(b/176884057): Find a supported alternative to gaussian_random_matrix,
# which became private in scikit-learn 0.24 and is likely to break.
# pylint: disable=g-import-not-at-top
try:
  from sklearn.random_projection import gaussian_random_matrix
except ImportError:
  from sklearn.random_projection import _gaussian_random_matrix as gaussian_random_matrix

import tensorflow as tf
import tensorflow_hub as hub
# pylint: enable=g-import-not-at-top

_RUNNER = 'DirectRunner'
_RANDOM_PROJECTION_FILENAME = 'random_projection.matrix'
_BATCH_SIZE = 1028

embed_fn = None


def generate_embeddings(items, module_url, random_projection_matrix=None):
  """Generates embeddings using a TF-Hub module.

  Args:
    items: The items to generate embedding for.
    module_url: The TF-Hub module url.
    random_projection_matrix: A numpy array of the random projection weights.

  Returns:
    item, embedding tuple.
  """

  global embed_fn
  if embed_fn is None:
    embed_fn = hub.load(module_url)
  embeddings = embed_fn(items).numpy()
  if random_projection_matrix is not None:
    embeddings = embeddings.dot(random_projection_matrix)
  return items, embeddings


def to_tf_example(entries):
  """Convert to tf example."""

  examples = []

  item_list, embedding_list = entries
  for i in range(len(item_list)):
    item = item_list[i]
    embedding = embedding_list[i]

    features = {
        'item':
            tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[item.encode('utf-8')])),
        'embedding':
            tf.train.Feature(
                float_list=tf.train.FloatList(value=embedding.tolist()))
    }

    example = tf.train.Example(features=tf.train.Features(
        feature=features)).SerializeToString(deterministic=True)

    examples.append(example)

  return examples


def generate_random_projection_weights(original_dim, projected_dim, output_dir):
  """Generates a Gaussian random projection weights matrix."""

  random_projection_matrix = None
  if projected_dim and original_dim > projected_dim:
    random_projection_matrix = gaussian_random_matrix(
        n_components=projected_dim, n_features=original_dim).T
    print('A Gaussian random weight matrix was creates with shape of {}'.format(
        random_projection_matrix.shape))
    print('Storing random projection matrix to disk...')
    output_file_path = os.path.join(output_dir, _RANDOM_PROJECTION_FILENAME)
    with open(output_file_path, 'wb') as handle:
      pickle.dump(
          random_projection_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Random projection matrix saved to disk.')

  return random_projection_matrix


def run(args):
  """Runs the embedding generation Beam pipeline."""

  if tf.io.gfile.exists(args.embed_output_dir):
    print('Removing embedding output directory...')
    tf.io.gfile.rmtree(args.embed_output_dir)
  print('Creating empty output directory...')
  tf.io.gfile.makedirs(args.embed_output_dir)

  options = beam.options.pipeline_options.PipelineOptions(**vars(args))

  original_dim = hub.load(args.module_url)(['']).shape[1]

  random_projection_matrix = generate_random_projection_weights(
      original_dim, args.projected_dim, args.embed_output_dir)

  print('Starting the Beam pipeline...')
  with beam.Pipeline(runner=_RUNNER, options=options) as pipeline:
    _ = (
        pipeline
        | 'Read sentences from files' >>
        beam.io.ReadFromText(file_pattern=args.data_file_pattern)
        | 'Batch elements' >> util.BatchElements(
            min_batch_size=_BATCH_SIZE / 2, max_batch_size=_BATCH_SIZE)
        | 'Generate embeddings' >> beam.Map(
            generate_embeddings, args.module_url, random_projection_matrix)
        | 'Encode to tf example' >> beam.FlatMap(to_tf_example)
        | 'Write to TFRecords files' >> beam.io.WriteToTFRecord(
            file_path_prefix='{}/emb'.format(args.embed_output_dir),
            file_name_suffix='.tfrecords')
    )

  print('Beam pipeline completed.')
