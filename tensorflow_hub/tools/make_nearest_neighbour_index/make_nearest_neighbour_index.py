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
"""Entry point to run the hub2ann tool."""

from absl import app
from absl import flags
import tensorflow as tf

from tensorflow_hub.tools.make_nearest_neighbour_index import embedding_generator as generator
from tensorflow_hub.tools.make_nearest_neighbour_index import index_builder as builder
from tensorflow_hub.tools.make_nearest_neighbour_index import similarity_finder as finder

# Embedding generator flags
flags.DEFINE_string(
    "data_file_pattern", None,
    "Path to data file(s) to generate embeddings for.")
flags.DEFINE_string(
    "module_url", None, "TF-Hub module to use. "
    "For more options, search https://tfhub.dev.")
flags.DEFINE_integer(
    "projected_dim", None,
    "The desired target dimension to project the embedding to. "
    "If specified, random projection will be uses.")
flags.DEFINE_string(
    "embed_output_dir", None,
    "The directory to store the generated embedding files to. "
    "This can  be a local or a GCS location.")

# index builder parameters
flags.DEFINE_integer(
    "num_trees", 100,
    "The number of trees to build the ANN index. Default is 100. "
    "For more details, refer to https://github.com/spotify/annoy.")
flags.DEFINE_string(
    "index_output_dir", None,
    "The directory to store the created index and mapping files. "
    "This can be a local or GCS location.")

# similarity matching parameters
flags.DEFINE_integer(
    "num_matches", 10,
    "The number of similar matches to retrieve from the ANN index. "
    "Default is 10.")

FLAGS = flags.FLAGS


def validate_args(args):
  """Validates the command line arguments specified by the user."""

  if len(args) < 2 or args[1] not in ["generate", "build", "e2e", "query"]:
    raise ValueError("You need to specify one of four operations: "
                     "generate | build | e2e | query")

  def _validate_generate_args():
    """Validates generate operation args."""
    if not FLAGS.data_file_pattern:
      raise ValueError(
          "You must provide --data_file_pattern to generate embeddings for.")
    if not FLAGS.module_url:
      raise ValueError(
          "You must provide --module_url to use for embeddings generation.")
    if not FLAGS.embed_output_dir:
      raise ValueError(
          "You must provide --embed_output_dir to store the embedding files.")
    if FLAGS.projected_dim and FLAGS.projected_dim < 1:
      raise ValueError("--projected_dim must be a positive integer value.")

  def _validate_build_args(e2e=False):
    """Validates build operation args."""
    if not FLAGS.embed_output_dir and not e2e:
      raise ValueError(
          "You must provide --embed_output_dir of the embeddings"
          "to build the ANN index for.")
    if not FLAGS.index_output_dir:
      raise ValueError(
          "You must provide --index_output_dir to store the index files.")
    if not FLAGS.num_trees or FLAGS.num_trees < 1:
      raise ValueError(
          "You must provide --num_trees as a positive integer value.")

  def _validate_query_args():
    if not FLAGS.module_url:
      raise ValueError("You must provide --module_url to use for query.")
    if not FLAGS.index_output_dir:
      raise ValueError("You must provide --index_output_dir to use for query.")

  operation = args[1]
  if operation == "generate":
    _validate_generate_args()
  elif operation == "build":
    _validate_build_args()
  elif operation == "e2e":
    _validate_generate_args()
    _validate_build_args(True)
  else:
    _validate_query_args()

  return operation


def _ensure_tf2():
  """Ensure running with TensorFlow 2 behavior.

  This function is safe to call even before flags have been parsed.

  Raises:
    ImportError: If tensorflow is too old for proper TF2 behavior.
  """
  print("Running with tensorflow %s (git version %s)",
        tf.__version__, tf.__git_version__)
  if tf.__version__.startswith("1."):
    if tf.__git_version__ == "unknown":  # For internal testing use.
      try:
        tf.compat.v1.enable_v2_behavior()
        return
      except AttributeError:
        pass  # Fail below for missing enabler function.
    raise ImportError("Sorry, this program needs TensorFlow 2.")


def main(args):
  """Entry point main function."""

  operation = validate_args(args)
  print("Selected operation: {}".format(operation))

  if operation == "generate":
    print("Generating embeddings...")
    generator.run(FLAGS)
    print("Embedding generation completed.")

  elif operation == "build":
    print("Building ANN index...")
    builder.run(FLAGS)
    print("Building ANN index completed.")

  elif operation == "e2e":
    print("Generating embeddings and building ANN index...")
    generator.run(FLAGS)
    print("Embedding generation completed.")
    if FLAGS.projected_dim:
      FLAGS.dimensions = FLAGS.projected_dim

    builder.run(FLAGS)
    print("Building ANN index completed.")

  else:
    print("Querying the ANN index...")
    similarity_finder = finder.load(FLAGS)
    num_matches = FLAGS.num_matches
    while True:
      print("Enter your query: ", end="")
      query = str(input())
      similar_items = similarity_finder.find_similar_items(query, num_matches)
      print("Results:")
      print("=========")
      for item in similar_items:
        print(item)


if __name__ == "__main__":
  _ensure_tf2()
  app.run(main)
