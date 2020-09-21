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
"""Tool to rank modules to use in a downstream classification task."""

from absl import app
from absl import flags
import numpy as np
import pandas as pd
import tensorflow.compat.v2 as tf

from tensorflow_hub.tools.module_search import utils

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset", None,
                    "Specification of a dataset. E.g. use `cifar10#1000` to "
                    "perform search using 1000 examples from tfds `cifar10` "
                    "dataset.")

flags.DEFINE_multi_string("module", None, "Module to consider in the search")

flags.DEFINE_string("module_list", None,
    "Path to text file with a module per line to be considered in the search."
    "Empty lines and lines starting with # are ignored")


def load_data(data_spec):
  return utils.load_data(**data_spec)


def load_raw_features(data_spec):
  data = load_data(data_spec=data_spec)
  return data.map(lambda x: tf.image.resize(x["image"], (224, 224)))


def load_labels(data_spec):
  data = load_data(data_spec=data_spec)
  return np.array([x for x in data.map(lambda x: x["label"])])


def compute_embeddings(module_spec, data_spec):
  raw_features = load_raw_features(data_spec=data_spec)
  embedding_fn = utils.load_embedding_fn(
      module=module_spec)
  outputs = []
  for batch in raw_features.batch(10):
    outputs.extend(embedding_fn(batch))
  return np.array(outputs)


def compute_score(module_spec, data_spec):
  embeddings = compute_embeddings(module_spec=module_spec,
                                  data_spec=data_spec)
  distances = utils.compute_distance_matrix_loo(embeddings)
  labels = load_labels(data_spec=data_spec)
  error_rate = utils.knn_errorrate_loo(distances, labels, k=1)
  return np.array(error_rate)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if not FLAGS.dataset:
    raise app.UsageError("--dataset is a required argument.")

  module_list = []
  if FLAGS.module:
    module_list.extend(FLAGS.module)

  if FLAGS.module_list:
    with tf.io.gfile.GFile(FLAGS.module_list) as f:
      lines = f.read().split("\n")
      module_list.extend([l for l in lines if l and not l.startswith("#")])

  if not module_list:
    raise app.UsageError(
        "Use --module or --module_list to define which modules to search.")

  ds_sections = FLAGS.dataset.split("#")
  dataset = ds_sections[0]
  train_examples = int(ds_sections[1]) if len(ds_sections) != 1 else None
  data_spec = {
    "dataset": dataset,
    "split": "train",
    "num_examples": train_examples,
  }

  results = []
  for module in module_list:
    results.append((
        module, data_spec,
        compute_score(module_spec=module, data_spec=data_spec)))

  df = pd.DataFrame(results, columns=["module", "data", "1nn"])
  df = df.filter(["module", "1nn"])
  df.sort_values(["1nn"])
  df.reset_index(drop=True)
  df.set_index("module")

  with pd.option_context(
      "display.max_rows", None,
      "display.max_columns", None,
      "display.precision", 3,
      "max_colwidth", -1,  # Don't truncate columns (e.g. module name).
      "display.expand_frame_repr", False,  # Don't wrap output.
  ):
    print("# Module ranking for %s" % data_spec)
    print(df)


if __name__ == "__main__":
  app.run(main)
