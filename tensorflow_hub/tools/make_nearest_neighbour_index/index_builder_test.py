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
"""Tests tensorflow_hub.tools.make_nearest_neighbour_index.index_builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl import flags
import tensorflow as tf
from tensorflow_hub.tools.make_nearest_neighbour_index import index_builder
# resources dependency

MNNI_FOLDER = ("org_tensorflow_hub/tools/"
               "make_nearest_neighbour_index/")

FLAGS = flags.FLAGS

flags.DEFINE_integer("embed_output_dir", None, "")
flags.DEFINE_integer("num_trees", 10, "")
flags.DEFINE_string("index_output_dir", None, "")


def _get_resource(dirname, filename):
  return os.path.join(os.path.dirname(__file__), filename)


class IndexBuilderTest(tf.test.TestCase):

  def setUp(self):  # pylint: disable=g-missing-super-call
    # Create run parameters
    FLAGS.embed_output_dir = _get_resource(MNNI_FOLDER, "test_data/embeds/")
    FLAGS.index_output_dir = os.path.join(self.get_temp_dir(), "index")

  def test_run(self):
    # Make sure we don't test for pre-existing files.
    self.assertFalse(os.path.isfile(FLAGS.index_output_dir))

    # Run index_builder
    index_builder.run(FLAGS)

    # Make sure that the index directory is created.
    self.assertTrue(os.path.exists(FLAGS.index_output_dir))
    # Make sure that the index file is created.
    expected_index = os.path.join(FLAGS.index_output_dir, "ann.index")
    self.assertTrue(os.path.isfile(expected_index))
    # Make sure that the mapping file is created.
    expected_mapping_file = os.path.join(FLAGS.index_output_dir,
                                         "ann.index.mapping")
    self.assertTrue(os.path.isfile(expected_mapping_file))
    # Make sure that the random prjection file is created.
    expected_projection_matrix_file = os.path.join(FLAGS.index_output_dir,
                                                   "random_projection.matrix")
    self.assertTrue(os.path.isfile(expected_projection_matrix_file))


def _ensure_tf2():
  """Ensure running with TensorFlow 2 behavior.

  This function is safe to call even before flags have been parsed.

  Raises:
    ImportError: If tensorflow is too old for proper TF2 behavior.
  """
  print("Running with tensorflow %s (git version %s)", tf.__version__,
        tf.__git_version__)
  if tf.__version__.startswith("1."):
    if tf.__git_version__ == "unknown":  # For internal testing use.
      try:
        tf.compat.v1.enable_v2_behavior()
        return
      except AttributeError:
        pass  # Fail below for missing enabler function.
    raise ImportError("Sorry, this program needs TensorFlow 2.")


if __name__ == "__main__":
  try:
    _ensure_tf2()
  except ImportError as e:
    print("Skipping tests:", str(e))
    sys.exit(0)
  tf.test.main()
