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
"""MetaGraph lib provides utilities to manipulate MetaGraphDefs.

This is an internal Hub utility and not part of the public API.
"""

import re

from absl import logging
import tensorflow as tf


def prepend_name_scope(name, import_scope):
  """Prepends name scope to a name."""
  # Based on tensorflow/python/framework/ops.py implementation.
  if import_scope:
    try:
      str_to_replace = r"([\^]|loc:@|^)(.*)"
      return re.sub(str_to_replace, r"\1" + import_scope + r"/\2",
                    tf.compat.as_str_any(name))
    except TypeError as e:
      # If the name is not of a type we can process, simply return it.
      logging.warning(e)
      return name
  else:
    return name


def prefix_shared_name_attributes(meta_graph, absolute_import_scope):
  """In-place prefixes shared_name attributes of nodes."""
  shared_name_attr = "shared_name"
  for node in meta_graph.graph_def.node:
    shared_name_value = node.attr.get(shared_name_attr, None)
    if shared_name_value and shared_name_value.HasField("s"):
      if shared_name_value.s:
        node.attr[shared_name_attr].s = tf.compat.as_bytes(
            prepend_name_scope(
                shared_name_value.s, import_scope=absolute_import_scope))


def mark_backward(output_tensor, used_node_names):
  """Function to propagate backwards in the graph and mark nodes as used.

  Traverses recursively through the graph from the end tensor, through the op
  that generates the tensor, and then to the input tensors that feed the op.
  Nodes encountered are stored in used_node_names.

  Args:
    output_tensor: A Tensor which we start the propagation.
    used_node_names: A list of strings, stores the name of nodes we've marked as
      visited.
  """
  op = output_tensor.op
  if op.name in used_node_names:
    return
  used_node_names.add(op.name)
  for input_tensor in op.inputs:
    mark_backward(input_tensor, used_node_names)
  for control_input_op in op.control_inputs:
    used_node_names.add(control_input_op.name)
    for input_tensor in control_input_op.inputs:
      mark_backward(input_tensor, used_node_names)


def prune_unused_nodes(meta_graph, signature_def):
  """Function to prune unused ops given a signature def.

  This function does a graph traversal through from all outputs as
  defined in the signature_def to collect all used nodes. Then, any
  nodes which are unused can be discarded. This is useful for graph which are
  executing eagerly or on TPUs.

  Args:
    meta_graph: The input/output MetaGraphDef for which we wish to prune.
   signature_def: A SignatureDef which specifies the outputs from which we wish
     to start graph traversal.
  """
  # Instantiate a temporary empty graph so that we have access to Graph API
  # and import the meta_graph.
  graph = tf.compat.v1.Graph()
  with graph.as_default():
    tf.compat.v1.train.import_meta_graph(meta_graph, input_map={},
                                         import_scope="")
    # Traverse from all outputs and mark all nodes.
    used_node_names = set()
    for _, tensor_def in signature_def.outputs.items():
      output_tensor = graph.get_tensor_by_name(tensor_def.name)
      mark_backward(output_tensor, used_node_names)
    # Filter out all nodes in the meta_graph that are not used.
    node_filter_in_list = []
    for node in meta_graph.graph_def.node:
      # Make a special exception for VarHandleOp. Removing VarhandleOps
      # will make the graph not importable as they often leave nodes hanging.
      # These will be disconnected through the feedmap when importing the
      # metagraph.
      if node.name in used_node_names or node.op == "VarHandleOp":
        node_filter_in_list.append(node)
    del meta_graph.graph_def.node[:]
    meta_graph.graph_def.node.extend(node_filter_in_list)
  del graph


def prune_feed_map(meta_graph, feed_map):
  """Function to prune the feedmap of nodes which no longer exist."""
  node_names = [x.name + ":0" for x in meta_graph.graph_def.node]
  keys_to_delete = []
  for k, _ in feed_map.items():
    if k not in node_names:
      keys_to_delete.append(k)
  for k in keys_to_delete:
    del feed_map[k]


def filter_collections(meta_graph, collections):
  collections = frozenset(collections)
  for name in list(meta_graph.collection_def.keys()):
    if name not in collections:
      del meta_graph.collection_def[name]
