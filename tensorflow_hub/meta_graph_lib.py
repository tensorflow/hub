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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

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
      tf.logging.warning(e)
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


def filter_collections(meta_graph, collections):
  collections = frozenset(collections)
  for name in list(meta_graph.collection_def.keys()):
    if name not in collections:
      del meta_graph.collection_def[name]
