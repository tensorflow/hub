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
"""Tests tensorflow_hub.tools.make_nearest_neighbour_index.embedding_generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl import flags
import tensorflow as tf
from tensorflow_hub.tools.make_nearest_neighbour_index import embedding_generator
# resources dependency

MNNI_FOLDER = ("org_tensorflow_hub/tools/"
               "make_nearest_neighbour_index/")

flags.DEFINE_string("data_file_pattern", None, "")
flags.DEFINE_string("module_url", None, "")
flags.DEFINE_integer("projected_dim", None, "")
flags.DEFINE_string("embed_output_dir", None, "")

FLAGS = flags.FLAGS


def _get_resource(dirname, filename):
  return os.path.join(os.path.dirname(__file__), filename)


class EmbeddingGeneratorTest(tf.test.TestCase):

  def setUp(self):  # pylint: disable=g-missing-super-call
    # create run parameters
    # FLAGS.data_file_pattern = _get_resource(
    # MNNI_FOLDER, "test_data/data/titles.txt")
    FLAGS.data_file_pattern = _get_resource(MNNI_FOLDER,
                                            "test_data/data/titles.txt")

    FLAGS.module_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
    FLAGS.embed_output_dir = os.path.join(self.get_temp_dir(), "embeds")

  def test_run(self):
    FLAGS.projected_dim = None

    # Make sure we don't test for pre-existing files.
    self.assertFalse(os.path.isfile(FLAGS.embed_output_dir))

    # Run embedding_generator
    embedding_generator.run(FLAGS)

    # Make sure that the embedding directory is created.
    self.assertTrue(os.path.exists(FLAGS.embed_output_dir))
    # Make sure that the embedding file is created.
    expected_embedding_file = os.path.join(FLAGS.embed_output_dir,
                                           "emb-00000-of-00001.tfrecords")
    self.assertTrue(os.path.isfile(expected_embedding_file))

  def test_run_with_projection(self):
    FLAGS.projected_dim = 64

    # Make sure we don't test for pre-existing files.
    self.assertFalse(os.path.isfile(FLAGS.embed_output_dir))

    # Run embedding_generator
    embedding_generator.run(FLAGS)

    # Make sure that the embedding directory is created.
    self.assertTrue(os.path.exists(FLAGS.embed_output_dir))
    # Make sure that the embedding file is created.
    expected_embedding_file = os.path.join(FLAGS.embed_output_dir,
                                           "emb-00000-of-00001.tfrecords")
    self.assertTrue(os.path.isfile(expected_embedding_file))
    # Make sure that the random prjection file is created.
    expected_projection_matrix_file = os.path.join(FLAGS.embed_output_dir,
                                                   "random_projection.matrix")
    self.assertTrue(os.path.isfile(expected_projection_matrix_file))


def _ensure_tf2():
  """Ensure running with TensorFlow 2 behavior.

  This function is safe to call even before flags have been parsed.

  Raises:
    ImportError: If tensorflow is too old for proper TF2 behavior.
  """
  print("Running with tensorflow %s", tf.__version__)
  if not tf.executing_eagerly():
    raise ImportError("Sorry, this program needs TensorFlow 2.")


if __name__ == "__main__":
  try:
    _ensure_tf2()
  except ImportError as e:
    print("Skipping tests:", str(e))
    sys.exit(0)
  tf.test.main()
