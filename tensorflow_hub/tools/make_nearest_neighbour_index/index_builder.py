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
"""Builds approximate nearest neighbor index for embeddings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import shutil

import annoy
import tensorflow as tf

_INDEX_FILENAME = 'ann.index'
_MAPPING_FILENAME = 'ann.index.mapping'
_RANDOM_PROJECTION_FILENAME = 'random_projection.matrix'
_METRIC = 'angular'


def _parse_example(example):
  """Parse TF Example."""

  # Create a description of the features in the tfrecords.
  feature_description = {
      'item': tf.io.FixedLenFeature([], tf.string),
      'embedding': tf.io.VarLenFeature(tf.float32)
  }
  # Parse the input `tf.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example, feature_description)


def _infer_dimensions(embed_file):
  """Infers the embedding vector size."""

  dimensions = None
  for record in tf.data.TFRecordDataset(embed_file).map(_parse_example):
    dimensions = record['embedding'].shape[0]
    break
  return dimensions


def run(args):
  """Runs the index building process."""

  embed_output_dir = args.embed_output_dir
  output_dir = args.index_output_dir
  num_trees = args.num_trees
  index_file_path = os.path.join(output_dir, _INDEX_FILENAME)
  mapping_file_path = os.path.join(output_dir, _MAPPING_FILENAME)

  if tf.io.gfile.exists(output_dir):
    print('Index output directory...')
    tf.io.gfile.rmtree(output_dir)
  print('Creating empty output directory...')
  tf.io.gfile.makedirs(output_dir)

  embed_files = tf.io.gfile.glob(os.path.join(embed_output_dir, '*.tfrecords'))
  num_files = len(embed_files)
  print('Found {} embedding file(s).'.format(num_files))

  dimensions = _infer_dimensions(embed_files[0])
  print('Embedding size: {}'.format(dimensions))

  annoy_index = annoy.AnnoyIndex(dimensions, metric=_METRIC)

  # Mapping between the item and its identifier in the index
  mapping = {}

  item_counter = 0
  for i, embed_file in enumerate(embed_files):
    print('Loading embeddings in file {} of {}...'.format(
        i + 1, num_files))
    dataset = tf.data.TFRecordDataset(embed_file)
    for record in dataset.map(_parse_example):
      item = record['item'].numpy().decode('utf-8')
      embedding = record['embedding'].values.numpy()
      mapping[item_counter] = item
      annoy_index.add_item(item_counter, embedding)
      item_counter += 1
      if item_counter % 200000 == 0:
        print('{} items loaded to the index'.format(item_counter))

  print('A total of {} items added to the index'.format(item_counter))

  print('Building the index with {} trees...'.format(num_trees))
  annoy_index.build(n_trees=num_trees)
  print('Index is successfully built.')

  print('Saving index to disk...')
  annoy_index.save(index_file_path)
  print('Index is saved to disk. File size: {} GB'.format(
      round(os.path.getsize(index_file_path) / float(1024**3), 2)))
  annoy_index.unload()

  print('Saving mapping to disk...')
  with open(mapping_file_path, 'wb') as handle:
    pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print('Mapping is saved to disk. File size: {} MB'.format(
      round(os.path.getsize(mapping_file_path) / float(1024**2), 2)))

  random_projection_file_path = os.path.join(
      args.embed_output_dir, _RANDOM_PROJECTION_FILENAME)
  if os.path.exists(random_projection_file_path):
    shutil.copy(
        random_projection_file_path, os.path.join(
            args.index_output_dir, _RANDOM_PROJECTION_FILENAME))
    print('Random projection matrix file copies to index output directory.')
