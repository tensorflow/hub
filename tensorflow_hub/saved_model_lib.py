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
"""SavedModel lib provides a way to read and write SavedModels.

This is an internal Hub utility and not part of the public API.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re

import tensorflow as tf
from tensorflow_hub import module_attachment_pb2
from tensorflow_hub import tf_utils

from google.protobuf import message
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2

# A collection of pairs (key: string, definition : SignatureDef) used internally
# to propagate signatures defined in a Graph to SavedModel. The collection key
# is a tuple (not a string) in order to make it invisible from user apis such
# as `get_all_collection_keys()` and manual exporting to meta_graphs.
_SIGNATURE_COLLECTION = ("__saved_model_lib_signatures",)

# A collection of ModuleAttachment protos is used internally to collect
# the (key, value) pairs passed to attach_message() calls from the module_fn.
# As above, it gets a non-string name to make it invisible within module_fn.
_ATTACHMENT_COLLECTION_INTERNAL = ("__hub_module_attachments",)
# The ModuleAttachment protos are stored in SavedModel.meta_graphs (but never
# in tf.Graphs) as CollectionDef.bytes_list under this key.
ATTACHMENT_COLLECTION_SAVED = "hub_module_attachments"


def get_variables_path(export_dir):
  """Returns the path for storing variables checkpoints."""
  return os.path.join(
      tf.compat.as_bytes(export_dir),
      tf.compat.as_bytes(tf.saved_model.constants.VARIABLES_DIRECTORY),
      tf.compat.as_bytes(tf.saved_model.constants.VARIABLES_FILENAME))


def _get_assets_dir(export_dir):
  return os.path.join(
      tf.compat.as_bytes(export_dir),
      tf.compat.as_bytes(tf.saved_model.constants.ASSETS_DIRECTORY))


def _get_asset_filename(export_dir, asset_filename):
  assets_dir = _get_assets_dir(export_dir)
  filename = os.path.join(
      tf.compat.as_bytes(assets_dir),
      tf.compat.as_bytes(asset_filename))
  if not tf_utils.absolute_path(filename).startswith(
      tf_utils.absolute_path(assets_dir)):
    raise ValueError(
        "Asset filename (%s) points outside assets_dir" % asset_filename)
  return filename


def _get_saved_model_proto_path(export_dir):
  return os.path.join(
      tf.compat.as_bytes(export_dir),
      tf.compat.as_bytes(tf.saved_model.constants.SAVED_MODEL_FILENAME_PB))


def _get_node_name_from_tensor(tensor_name):
  """tensor_name must have format node_name:output_number. Returns node_name."""
  result = re.match(r"([^:]*):\d+$", tensor_name)
  if not result:
    raise ValueError(
        "Unexpected format for tensor name. Expected node_name:output_number. "
        "Got %r" % tensor_name)
  return result.group(1)


def add_signature(key, inputs, outputs):
  """Adds a signature to current graph.

  Args:
    key: Signature key as a string.
    inputs: Signature inputs as a map from string to Tensor or SparseTensor.
    outputs: Signature outputs as a map from string to Tensor or SparseTensor.
      (Recall that a Variable is not a Tensor, but Variable.value() is.)

  Raises:
    TypeError: if the arguments have the wrong types.
  """
  _check_dict_maps_to_tensors_or_sparse_tensors(inputs)
  _check_dict_maps_to_tensors_or_sparse_tensors(outputs)
  input_info = {
      input_name: tf.saved_model.utils.build_tensor_info(tensor)
      for input_name, tensor in inputs.items()
  }
  output_info = {
      output_name: tf.saved_model.utils.build_tensor_info(tensor)
      for output_name, tensor in outputs.items()
  }
  signature = tf.saved_model.signature_def_utils.build_signature_def(
      input_info, output_info)
  tf.add_to_collection(_SIGNATURE_COLLECTION, (key, signature))


def _check_dict_maps_to_tensors_or_sparse_tensors(tensor_map):
  for key, value in tensor_map.items():
    if not isinstance(value, (tf.Tensor, tf.SparseTensor)):
      raise TypeError(
          "Value for key '%s' should be a Tensor or SparseTensor object, found"
          " %s." % (key, type(value)))


def _export_signatures(meta_graph):
  """Exports signatures from current graph into a MetaGraphDef."""
  named_signatures = tf.get_collection(_SIGNATURE_COLLECTION)
  if not named_signatures:
    raise ValueError("No signatures present. Please call hub.add_signature(...)"
                     "at least once in the module_fn.")
  for key, signature in named_signatures:
    meta_graph.signature_def[key].CopyFrom(signature)


def attach_bytes(key, the_bytes):
  """Adds a ModuleAttachment to the current graph.

  Args:
    key: A string with the unique key of the attachment.
    the_bytes: A bytes object with the serialized attachment.
  """
  tf.add_to_collection(
      _ATTACHMENT_COLLECTION_INTERNAL,
      module_attachment_pb2.ModuleAttachment(key=key, value=the_bytes))


def _export_module_attachments(meta_graph):
  """Exports ModuleAttachments from the current tf.Graph into `meta_graph`."""
  added_attachments = tf.get_collection(_ATTACHMENT_COLLECTION_INTERNAL)
  if not added_attachments: return  # Don't touch `meta_graph`.
  unique_attachments = collections.OrderedDict(  # Avoid indeterminism.
      (attachment.key, attachment)
      for attachment in added_attachments)
  meta_graph.collection_def[ATTACHMENT_COLLECTION_SAVED].bytes_list.value[:] = [
      attachment.SerializeToString()
      for attachment in unique_attachments.values()]


def get_attached_bytes_map(meta_graph):
  """Returns the dict of ModuleAttachments stored in `meta_graph`.

  Args:
    meta_graph: A MetaGraphDef, as built by SavedModelHandler.add_graph_copy()
      from some graph.

  Returns:
    A dict, containing the `(key, bytes)` items passed to `attach_bytes()`
    when the graph had been built.

  Raises:
    ValueError: if `meta-graph` is malformed.
  """
  result = {}
  if ATTACHMENT_COLLECTION_SAVED not in meta_graph.collection_def:
    return result
  collection_def = meta_graph.collection_def[ATTACHMENT_COLLECTION_SAVED]
  if collection_def.WhichOneof("kind") != "bytes_list":
    raise ValueError(
        "Internal CollectionDef for attached messages has kind %s, "
        "expected bytes_list" % collection_def.WhichOneof("kind"))
  attachment = module_attachment_pb2.ModuleAttachment()
  for value in collection_def.bytes_list.value:
    attachment.ParseFromString(value)
    result[attachment.key] = attachment.value  # Immutable; needs no copy.
  return result


def _export_tags(meta_graph, tags):
  """Exports tags into a MetaGraphDef."""
  if tags is not None:
    meta_graph.meta_info_def.tags.extend(tags)


def _check_asset_node_def(node_def):
  """Raises TypeError if `node_def` does not match the expectations."""
  if node_def.op != "Const":
    raise TypeError("Asset node must be of type constant.")
  if tf.as_dtype(node_def.attr["dtype"].type) != tf.string:
    raise TypeError("Asset node must be of dtype string.")
  if len(node_def.attr["value"].tensor.string_val) != 1:
    raise TypeError("Asset node must be a scalar.")


def _merge_assets_key_collection(saved_model_proto, path):
  """Merges the ASSETS_KEY collection into the GraphDefs in saved_model_proto.

  Removes the ASSETS_KEY collection from the GraphDefs in the SavedModel and
  modifies nodes with the assets filenames to point to the assets in `path`.
  After this transformation, the SavedModel GraphDefs can be used without
  feeding asset tensors.

  Args:
    saved_model_proto: SavedModel proto to be modified.
    path: path where the SavedModel is being loaded from.
  """
  for meta_graph in saved_model_proto.meta_graphs:
    node_asset_map = {}
    if tf.saved_model.constants.ASSETS_KEY in meta_graph.collection_def:
      assets_any_proto = meta_graph.collection_def[
          tf.saved_model.constants.ASSETS_KEY].any_list.value
      for asset_any_proto in assets_any_proto:
        asset_proto = meta_graph_pb2.AssetFileDef()
        asset_any_proto.Unpack(asset_proto)
        asset_filename = _get_asset_filename(path, asset_proto.filename)
        node_asset_map[_get_node_name_from_tensor(
            asset_proto.tensor_info.name)] = asset_filename
      del meta_graph.collection_def[tf.saved_model.constants.ASSETS_KEY]

    for node in meta_graph.graph_def.node:
      asset_filepath = node_asset_map.get(node.name)
      if asset_filepath:
        _check_asset_node_def(node)
        node.attr["value"].tensor.string_val[0] = asset_filepath


def _make_assets_key_collection(saved_model_proto, export_path):
  """Creates an ASSETS_KEY collection in the GraphDefs in saved_model_proto.

  Adds an ASSETS_KEY collection to the GraphDefs in the SavedModel and returns
  a map from original asset filename to filename when exporting the SavedModel
  to `export_path`.

  This is roughly the inverse operation of `_merge_assets_key_collection`.

  Args:
    saved_model_proto: SavedModel proto to be modified.
    export_path: string with path where the saved_model_proto will be exported.

  Returns:
    A map from original asset filename to asset filename when exporting the
    SavedModel to path.

  Raises:
    ValueError: on unsuported/unexpected SavedModel.
  """
  asset_filenames = {}
  used_asset_filenames = set()

  def _make_asset_filename(original_filename):
    """Returns the asset filename to use for the file."""
    if original_filename in asset_filenames:
      return asset_filenames[original_filename]

    basename = os.path.basename(original_filename)
    suggestion = basename
    index = 0
    while suggestion in used_asset_filenames:
      suggestion = "%s%d" % (basename, index)
      index += 1
    asset_filenames[original_filename] = suggestion
    used_asset_filenames.add(suggestion)
    return suggestion

  for meta_graph in saved_model_proto.meta_graphs:
    collection_def = meta_graph.collection_def.get(
        tf.GraphKeys.ASSET_FILEPATHS)

    if collection_def is None:
      continue
    if collection_def.WhichOneof("kind") != "node_list":
      raise ValueError(
          "MetaGraph collection ASSET_FILEPATHS is not a list of tensors.")

    for tensor in collection_def.node_list.value:
      if not tensor.endswith(":0"):
        raise ValueError("Unexpected tensor in ASSET_FILEPATHS collection.")

    asset_nodes = set([
        _get_node_name_from_tensor(tensor)
        for tensor in collection_def.node_list.value
    ])

    tensor_filename_map = {}
    for node in meta_graph.graph_def.node:
      if node.name in asset_nodes:
        _check_asset_node_def(node)
        filename = node.attr["value"].tensor.string_val[0]
        tensor_filename_map[node.name + ":0"] = filename
        # Clear value to avoid leaking the original path.
        node.attr["value"].tensor.string_val[0] = (
            tf.compat.as_bytes("SAVEDMODEL-ASSET"))

    if tensor_filename_map:
      assets_key_collection = meta_graph.collection_def[
          tf.saved_model.constants.ASSETS_KEY]

      for tensor, filename in sorted(tensor_filename_map.items()):
        asset_proto = meta_graph_pb2.AssetFileDef()
        asset_proto.filename = _make_asset_filename(filename)
        asset_proto.tensor_info.name = tensor
        assets_key_collection.any_list.value.add().Pack(asset_proto)

  return {
      original_filename: _get_asset_filename(export_path, asset_filename)
      for original_filename, asset_filename in asset_filenames.items()
  }


class SavedModelHandler(object):
  """SavedModelHandler helps using SavedModel disk format.

  Note: This is a lower level interface than most users need. See SavedModel
  Builder/Loader API for an higher-level API centered around exporting and
  loading Sessions.

  A SavedModel disk format represents a collection of Graphs. To allow these
  graphs to be easy to manipulate, SavedModel extends Graphs with tags and
  signatures. Additionally it packages graphs, assets and variable checkpoints
  into an hermetic directory that can be moved around.

  This class hides the implementation details of SavedModels, in particular
  related with assets and signatures.

  SavedModelHandler deals with assets by:
    - Only supporting asset files as constant ops added to ASSET_FILEPATHS
      collection.
    - Creating a ASSETS_KEY collection only when writing meta_graphs to disk so
      they are never visible to user.
    - Baking the ASSETS_KEY collection in the graphs when loading from disk as
      to hide that the assets point to the packaged assets.

  SavedModelHandler deals with signatures by:
    - Providing `add_signature` API that allows to declare signatures directly
      on a graph.
    - That API is supported by a collection that is not serialized, but instead
      is converted into the right fields of MetaGraphDef when writing and
      loading a SavedModel from disk.
  """

  def __init__(self):
    self._proto = saved_model_pb2.SavedModel()

  def add_graph_copy(self, graph, tags=None):
    """Adds a copy of Graph with the specified set of tags."""
    with graph.as_default():
      # Remove default attrs so that Modules created by a tensorflow version
      # with ops that have new attrs that are left to their default values can
      # still be loaded by older versions unware of those attributes.
      meta_graph = tf.train.export_meta_graph(strip_default_attrs=True)
      _export_tags(meta_graph, tags)
      _export_signatures(meta_graph)
      _export_module_attachments(meta_graph)
    self._proto.meta_graphs.extend([meta_graph])

  def add_meta_graph_copy(self, meta_graph):
    self._proto.meta_graphs.extend([meta_graph])

  def get_meta_graph_copy(self, tags=None):
    """Returns a copy of a MetaGraph with the identical set of tags."""
    meta_graph = self.get_meta_graph(tags)
    copy = tf.MetaGraphDef()
    copy.CopyFrom(meta_graph)
    return copy

  @property
  def meta_graphs(self):
    return iter(self._proto.meta_graphs)

  def get_tags(self):
    """Returns a list of set of tags."""
    return sorted([frozenset(meta_graph.meta_info_def.tags)
                   for meta_graph in self.meta_graphs])

  def get_attached_bytes_map(self, tags=None):
    return get_attached_bytes_map(self.get_meta_graph(tags))

  def export(self, path, variables_saver=None):
    """Exports to SavedModel directory.

    Args:
      path: path where to export the SavedModel to.
      variables_saver: lambda that receives a directory path where to
        export checkpoints of variables.
    """
    # Operate on a copy of self._proto since it needs to be modified.
    proto = saved_model_pb2.SavedModel()
    proto.CopyFrom(self._proto)
    assets_map = _make_assets_key_collection(proto, path)

    self._save_all_assets(path, assets_map)
    self._save_variables(path, variables_saver)
    self._save_proto(path, proto)

  def get_meta_graph(self, tags=None):
    """Returns the matching MetaGraphDef or raises KeyError."""
    matches = [meta_graph
               for meta_graph in self.meta_graphs
               if set(meta_graph.meta_info_def.tags) == set(tags or [])]
    if not matches:
      raise KeyError("SavedModelHandler has no graph with tags: %r" % tags)
    if len(matches) != 1:
      raise KeyError(
          "SavedModelHandler has multiple graphs with tags %r" % tags)
    return matches[0]

  def _save_all_assets(self, path, assets_map):
    assets_dir = _get_assets_dir(path)
    tf.gfile.MakeDirs(assets_dir)
    for source, destination in assets_map.items():
      tf.gfile.Copy(source, destination)

  def _save_variables(self, path, variables_saver):
    if variables_saver:
      variables_path = get_variables_path(path)
      variables_dir = os.path.dirname(variables_path)
      tf.gfile.MakeDirs(variables_dir)
      variables_saver(variables_path)

  def _save_proto(self, path, proto):
    proto_path = _get_saved_model_proto_path(path)
    tf.gfile.MakeDirs(os.path.dirname(proto_path))
    tf_utils.atomic_write_string_to_file(proto_path,
                                         proto.SerializeToString(),
                                         overwrite=True)


def _parse_saved_model(path):
  """Reads the savedmodel.pb file containing `SavedModel`."""
  # Based on tensorflow/python/saved_model/loader.py implementation.
  path_to_pb = _get_saved_model_proto_path(path)
  file_content = tf.gfile.Open(path_to_pb, "rb").read()
  saved_model = saved_model_pb2.SavedModel()
  try:
    saved_model.ParseFromString(file_content)
  except message.DecodeError as e:
    raise IOError("Cannot parse file %s: %s." % (path_to_pb, str(e)))
  return saved_model


def load(path):
  """Creates a SavedModelHandler from a SavedModel in `path`."""
  proto = _parse_saved_model(path)
  _merge_assets_key_collection(proto, path)
  handler = SavedModelHandler()
  handler._proto = proto  # pylint: disable=protected-access
  return handler
