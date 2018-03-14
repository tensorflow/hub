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
"""Creates a simple TF-Hub Module.

The module has a single default signature that computes a*x+b. Where 'a' and 'b'
are variables in the graph. Before export, the Module is "trained" by explicitly
setting those variables to the magic numbers that make it compute:

  0.5 * x + 2  # Half plus two.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_hub as hub


def half_plus_two():
  a = tf.get_variable("a", shape=[])
  b = tf.get_variable("b", shape=[])
  x = tf.placeholder(tf.float32)
  y = a*x + b
  hub.add_signature(inputs=x, outputs=y)


def export_module(path):
  spec = hub.create_module_spec(half_plus_two)

  with tf.Graph().as_default():
    module = hub.Module(spec)

    init_a = tf.assign(module.variable_map["a"], 0.5)
    init_b = tf.assign(module.variable_map["b"], 2.0)
    init_vars = tf.group([init_a, init_b])

    with tf.Session() as session:
      session.run(init_vars)
      module.export(path, session)


def main(argv):
  try:
    _, export_path, = argv
  except ValueError:
    raise ValueError("Usage: %s <export-path>" % argv[0])

  if tf.gfile.Exists(export_path):
    raise RuntimeError("Path %s already exists." % export_path)

  export_module(export_path)


if __name__ == "__main__":
  tf.app.run(main)
