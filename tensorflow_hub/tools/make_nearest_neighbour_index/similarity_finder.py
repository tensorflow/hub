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
"""Find similar items for a given query in the ANN index."""

import os
import pickle

import annoy
import tensorflow as tf
import tensorflow_hub as hub

_INDEX_FILENAME = 'ann.index'
_MAPPING_FILENAME = 'ann.index.mapping'
_RANDOM_PROJECTION_FILENAME = 'random_projection.matrix'
_METRIC = 'angular'


class SimilarityFinder(object):
  """Similarity finder class."""

  def __init__(
      self,
      module_url,
      index_file_path,
      mapping_file_path,
      dimensions,
      random_projection_matrix_file,
  ):

    # Load the TF-Hub module
    print('Loading the TF-Hub module...')
    self.embed_fn = hub.load(module_url)
    print('TF-hub module is loaded.')

    dimensions = self.embed_fn(['']).shape[1]

    self.random_projection_matrix = None
    if tf.io.gfile.exists(random_projection_matrix_file):
      with open(random_projection_matrix_file, 'rb') as handle:
        self.random_projection_matrix = pickle.load(handle)
      dimensions = self.random_projection_matrix.shape[1]

    self.index = annoy.AnnoyIndex(dimensions, metric=_METRIC)
    self.index.load(index_file_path, prefault=True)
    print('Annoy index is loaded.')
    with open(mapping_file_path, 'rb') as handle:
      self.mapping = pickle.load(handle)
    print('Mapping file is loaded.')

  def find_similar_items(self, query, num_matches=5):
    """Finds similar items to a given quey in the ANN index.

    Args:
      query: The query string
      num_matches: The number of similar items to retrieve.

    Returns:
      List of items.
    """

    query_embedding = self.embed_fn([query])[0].numpy()
    if self.random_projection_matrix is not None:
      query_embedding = query_embedding.dot(self.random_projection_matrix)
    ids = self.index.get_nns_by_vector(
        query_embedding, num_matches, search_k=-1, include_distances=False)
    items = [self.mapping[i] for i in ids]
    return items


def load(args):

  module_url = args.module_url
  index_file_path = os.path.join(args.index_output_dir, _INDEX_FILENAME)
  mapping_file_path = os.path.join(args.index_output_dir, _MAPPING_FILENAME)
  dimensions = args.dimensions
  random_projection_matrix_file = os.path.join(
      args.index_output_dir, _RANDOM_PROJECTION_FILENAME)

  return SimilarityFinder(
      module_url,
      index_file_path,
      mapping_file_path,
      dimensions,
      random_projection_matrix_file,
  )
