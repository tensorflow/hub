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
"""Module implementation that loads/exports in Hub SavedModel format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import tensorflow as tf

from tensorflow_hub import compressed_module_resolver
from tensorflow_hub import module_def_pb2
from tensorflow_hub import module_impl
from tensorflow_hub import module_spec
from tensorflow_hub import saved_model_lib
from tensorflow_hub import tensor_info
from tensorflow_hub import tf_utils

from tensorflow.core.protobuf import meta_graph_pb2

# TODO(b/72732111): Get this APIs or similar functionality to be public.
# They are needed to identify the "state-ops" in a graph and to load C
# registered ops into the python register for import_meta_graph to succeed
# without having to do "import library_that_register_missing_op".
# pylint: disable=g-bad-import-order
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python import pywrap_tensorflow as c_api
# pylint: enable=g-bad-import-order

_MODULE_PROTO_FILENAME_PB = "tfhub_module.pb"

_MODULE_V3_SUPPORTED_FEATURES = frozenset([])  # None yet.

_SUPPORTED_COLLECTIONS = set([
    # GLOBAL_VARIABLES, TRAINABLE_VARIABLES and MODEL_VARIABLES hold
    # tf.Variable objects saved in CollectionDef.bytes_list as serialized
    # VariableDef proto.
    tf.GraphKeys.GLOBAL_VARIABLES,
    tf.GraphKeys.TRAINABLE_VARIABLES,
    tf.GraphKeys.MODEL_VARIABLES,
    # This holds tf.Operation objects, saved in CollectionDef.node_list.
    tf.GraphKeys.TABLE_INITIALIZERS,
    # This holds tf.Tensor objects, saved in CollectionDef.node_list.
    tf.GraphKeys.UPDATE_OPS,
    # This holds tf.Tensor objects, saved in CollectionDef.node_list.
    # These are imported to help fine-tuning (unlike LOSSES, which the
    # importing model redefines from scratch).
    tf.GraphKeys.REGULARIZATION_LOSSES,
    # This holds constant tensors of type string.
    tf.GraphKeys.ASSET_FILEPATHS,
    # This holds serialized CondContextDef protos in CollectionDef.bytes_list.
    tf.GraphKeys.COND_CONTEXT,
    # This holds serialized WhileContextDef protos in CollectionDef.bytes_list.
    tf.GraphKeys.WHILE_CONTEXT,
])


def _get_module_proto_path(module_dir):
  return os.path.join(
      tf.compat.as_bytes(module_dir),
      tf.compat.as_bytes(_MODULE_PROTO_FILENAME_PB))


def load_module_spec(path):
  """Loads a ModuleSpec from the filesystem.

  Args:
    path: string describing the location of a module. There are several
          supported path encoding schemes:
          a) URL location specifying an archived module
            (e.g. http://domain/module.tgz)
          b) Any filesystem location of a module directory (e.g. /module_dir
             for a local filesystem). All filesystems implementations provided
             by Tensorflow are supported.

  Returns:
    A ModuleSpec.

  Raises:
    ValueError: on unexpected values in the module spec.
    tf.OpError: on file handling exceptions.
  """
  path = compressed_module_resolver.get_default().get_module_path(path)
  module_def_path = _get_module_proto_path(path)
  module_def_proto = module_def_pb2.ModuleDef()
  with tf.gfile.Open(module_def_path, "rb") as f:
    module_def_proto.ParseFromString(f.read())

  if module_def_proto.format != module_def_pb2.ModuleDef.FORMAT_V3:
    raise ValueError("Unsupported module def format: %r" %
                     module_def_proto.format)

  required_features = set(module_def_proto.required_features)
  unsupported_features = (required_features - _MODULE_V3_SUPPORTED_FEATURES)

  if unsupported_features:
    raise ValueError("Unsupported features: %r" % list(unsupported_features))

  saved_model_handler = saved_model_lib.load(path)
  checkpoint_filename = saved_model_lib.get_variables_path(path)
  return _ModuleSpec(saved_model_handler, checkpoint_filename)


def create_module_spec(module_fn, tags_and_args=None, drop_collections=None):
  """Creates a ModuleSpec from a function that builds the module's graph.

  The `module_fn` is called on a new graph (not the current one) to build the
  graph of the module and define its signatures via `hub.add_signature()`.
  Example:

  ```python
  # Define a text embedding module.
  def my_text_module_fn():
    text_input = tf.placeholder(dtype=tf.string, shape=[None])
    embeddings = compute_embedding(text)
    hub.add_signature(inputs=text_input, outputs=embeddings)
  ```

  See `add_signature()` for documentation on adding multiple input/output
  signatures.

  NOTE: In anticipation of future TF-versions, `module_fn` is called on a graph
  that uses resource variables by default. If you want old-style variables then
  you can use `with tf.variable_scope("", use_resource=False)` in `module_fn`.

  Multiple graph variants can be defined by using the `tags_and_args` argument.
  For example, the code:

  ```python
  hub.create_module_spec(
      module_fn,
      tags_and_args=[({"train"}, {"is_training":True}),
                     (set(), {"is_training":False})])
  ```

  calls `module_fn` twice, once as `module_fn(is_training=True)` and once as
  `module_fn(is_training=False)` to define the respective graph variants:
  for training with tags {"train"} and for inference with the empty set of tags.
  Using the empty set aligns the inference case with the default in
  Module.__init__().

  Args:
    module_fn: a function to build a graph for the Module.
    tags_and_args: Optional list of tuples (tags, kwargs) of tags and keyword
      args used to define graph variants. If omitted, it is interpreted as
      [set(), {}], meaning `module_fn` is called once with no args.
    drop_collections: list of collection to drop.

  Returns:
    A ModuleSpec.

  Raises:
    ValueError: if it fails to construct the ModuleSpec due to bad or
      unsupported values in the arguments or in the graphs constructed by
      `module_fn`.
  """
  if not drop_collections:
    drop_collections = []

  report_tags = True
  if not tags_and_args:
    tags_and_args = [(set(), {})]
    report_tags = False

  saved_model_handler = saved_model_lib.SavedModelHandler()
  for tags, args in tags_and_args:
    with tf.Graph().as_default() as graph:
      with tf.variable_scope("", use_resource=True):
        module_fn(**args)

      for collection_key in drop_collections:
        del tf.get_collection_ref(collection_key)[:]

    err = find_state_op_colocation_error(graph, tags if report_tags else None)
    if err: raise ValueError(err)
    saved_model_handler.add_graph_copy(graph, tags=tags)

  return _ModuleSpec(saved_model_handler, checkpoint_variables_path=None)


def add_signature(name=None, inputs=None, outputs=None):
  """Adds a signature to the module definition.

  NOTE: This must be called within a `module_fn` that is defining a Module.

  Args:
    name: Signature name as a string. If omitted, it is interpreted as 'default'
      and is the signature used when `Module.__call__` `signature` is not
      specified.
    inputs: A dict from input name to Tensor or SparseTensor to feed when
      applying the signature. If a single tensor is passed, it is interpreted
      as a dict with a single 'default' entry.
    outputs: A dict from output name to Tensor or SparseTensor to return from
      applying the signature. If a single tensor is passed, it is interpreted
      as a dict with a single 'default' entry.

  Raises:
    ValueError: if the arguments are invalid.
  """
  if not name:
    name = "default"
  if inputs is None:
    inputs = {}
  if outputs is None:
    outputs = {}
  if not isinstance(inputs, dict):
    inputs = {"default": inputs}
  if not isinstance(outputs, dict):
    outputs = {"default": outputs}
  err = find_signature_input_colocation_error(name, inputs)
  if err: raise ValueError(err)
  saved_model_lib.add_signature(name, inputs, outputs)


class _ModuleSpec(module_spec.ModuleSpec):
  """ModuleSpec for Hub's native Module format (backed by SavedModel)."""

  def __init__(self, saved_model_handler, checkpoint_variables_path):
    """Private constructor.

    Args:
      saved_model_handler: SavedModelHandler backing up this Module definition.
      checkpoint_variables_path: An optional string to the checkpoint where this
        Module variables are checkpointed. If given the variables initializers
        are overridden to load from it.

    Raises:
      ValueError: if SavedModel contains any unexpected value.
    """
    check_unique_tags(saved_model_handler.get_tags())
    check_collections_are_supported(saved_model_handler, _SUPPORTED_COLLECTIONS)
    self._saved_model_handler = saved_model_handler
    self._checkpoint_variables_path = checkpoint_variables_path

  def get_tags(self):
    return self._saved_model_handler.get_tags()

  def get_signature_names(self, tags=None):
    meta_graph = self._saved_model_handler.get_meta_graph(tags=tags)
    return list(meta_graph.signature_def.keys())

  def get_input_info_dict(self, signature=None, tags=None):
    signature_def = self._get_signature_def(signature, tags)
    return tensor_info.parse_tensor_info_map(signature_def.inputs)

  def get_output_info_dict(self, signature=None, tags=None):
    signature_def = self._get_signature_def(signature, tags)
    return tensor_info.parse_tensor_info_map(signature_def.outputs)

  def _get_signature_def(self, signature, tags):
    meta_graph = self._saved_model_handler.get_meta_graph(tags=tags)
    if signature is None:
      signature = "default"
    signature_def = meta_graph.signature_def.get(signature)
    if signature_def is None:
      raise ValueError("Signature %r is missing from meta graph." % signature)
    return signature_def

  def _create_impl(self, name, trainable, tags):
    meta_graph = self._saved_model_handler.get_meta_graph(tags=tags)
    return _ModuleImpl(
        spec=self,
        meta_graph=meta_graph,
        trainable=trainable,
        checkpoint_path=self._checkpoint_variables_path,
        name=name)

  def _export(self, path, variables_saver):
    """Internal.

    Args:
      path: string where to export the module to.
      variables_saver: an unary-function that writes the module variables
        checkpoint on the given path.
    """
    self._saved_model_handler.export(path, variables_saver=variables_saver)

    module_def_proto = module_def_pb2.ModuleDef()
    module_def_proto.format = module_def_pb2.ModuleDef.FORMAT_V3
    module_def_filename = _get_module_proto_path(path)
    tf_utils.atomic_write_string_to_file(
        module_def_filename,
        module_def_proto.SerializeToString(),
        overwrite=False)
    tf.logging.info("Exported TF-Hub module to: %s", path)


class _ModuleImpl(module_impl.ModuleImpl):
  """A Module instantiation backed by a MetaGraphDef."""

  def __init__(self, spec, meta_graph, trainable, checkpoint_path, name):
    """Private constructor.

    Args:
      spec: _ModuleSpec instance.
      meta_graph: MetaGraphDef to use
      trainable: whether module is trainable.
      checkpoint_path: None or a string to the variables checkpoints.
      name: variable and scope name where to instantiate the Module. Must be an
        unused name scope.
    """
    self._spec = spec
    self._graph = tf.get_default_graph()
    self._meta_graph = meta_graph
    self._trainable = trainable
    self._checkpoint_path = checkpoint_path

    register_ops_if_needed({
        op.name for op in self._meta_graph.meta_info_def.stripped_op_list.op})

    # Clear dependencies so Modules can be constructed from deep inside
    # functions that have dependencies active. Note that the dependencies
    # would be active when applying the Module signature, just not active
    # when creating the Module state. This use case has showed up in some
    # TPU training code.
    with tf.control_dependencies(None):
      variable_tensor_map, self._state_map = self._create_state_graph(name)
      self._variable_map = recover_partitioned_variable_map(
          get_node_map_from_tensor_map(variable_tensor_map)
      )
      if self._variable_map and self._checkpoint_path:
        tf.train.init_from_checkpoint(self._checkpoint_path, self._variable_map)

      # Build Saver so it can be used later on to export the variables.
      if self._variable_map:
        self._saver = tf.train.Saver(
            self._variable_map,
            sharded=True,
            write_version=tf.train.SaverDef.V2)
      else:
        self._saver = None

  def _create_state_graph(self, name):
    """Creates the graph nodes that hold the state of the Module.

    Args:
      name: name scope to create the state graph in.

    Returns:
      A tuple consisting of:
        variables_tensor_map: a map from tensor names in the original graph def
          to the created Variables objects.
        state_map: a map from tensors names in the original graph def to the
          instantiated tensors to be used as a state_map.
    """
    import_collections = [
        tf.GraphKeys.GLOBAL_VARIABLES,
        tf.GraphKeys.MODEL_VARIABLES,
        tf.GraphKeys.TABLE_INITIALIZERS,
        tf.GraphKeys.ASSET_FILEPATHS,  # Typically used to initialize tables.
        tf.GraphKeys.COND_CONTEXT,
        tf.GraphKeys.WHILE_CONTEXT,
    ]
    if self._trainable:
      # TODO(b/64049014): Import UPDATE_OPS which do not depend on inputs.
      # This is needed for proper interaction with tf.contrib.quantize.
      import_collections.extend([tf.GraphKeys.TRAINABLE_VARIABLES,
                                 tf.GraphKeys.REGULARIZATION_LOSSES])

    absolute_scope_name = self._graph.unique_name(name, mark_as_used=False)
    relative_scope_name = absolute_scope_name.split("/")[-1]
    assert relative_scope_name == name  # verify name scope was indeed unused.

    tf.train.import_meta_graph(
        adapted_meta_graph_for_import(self._meta_graph, absolute_scope_name),
        import_scope=relative_scope_name,
        restore_collections_predicate=(lambda key: key in import_collections))

    # Build a list from the variable name in the module definition to the actual
    # instantiated variables.
    variables_tensor_map = {}
    for var in tf.global_variables():
      if var.op.name.startswith(absolute_scope_name + "/"):
        variables_tensor_map[var.name[len(absolute_scope_name)+1:]] = var

    # Build a map of tensors to feed from the state-graph into subsequent
    # apply-graphs.
    def _get_tensor(tensor_name):
      return self._graph.get_tensor_by_name(
          prepend_name_scope(tensor_name, import_scope=absolute_scope_name))

    state_op_names = list_registered_stateful_ops_without_inputs()
    state_map = get_state_map(self._meta_graph, state_op_names, set(),
                              _get_tensor)

    return variables_tensor_map, state_map

  def create_apply_graph(self, signature, inputs, name):
    """See `ModuleImpl.create_apply_graph`."""
    signature_def = self._meta_graph.signature_def.get(signature)

    input_tensors = tensor_info.convert_to_input_tensors(
        signature_def.inputs, inputs)

    # Build a input map to feed when importing the apply-graph by augmenting the
    # state_map with the input args. This allows an input to override a tensor
    # from the state-graph.
    feed_map = dict(self._state_map)
    feed_map.update(
        tensor_info.build_input_map(signature_def.inputs, input_tensors))

    # Make state tensors enter the current context. This way the Module can be
    # applied inside a control flow structure such as a while_loop.
    control_flow = self._graph._get_control_flow_context()  # pylint: disable=protected-access
    if control_flow:
      for key, value in feed_map.items():
        feed_map[key] = control_flow.AddValue(value)

    # Don't mark the name as used at this point - import_scoped_meta_graph will
    # start using it.
    absolute_scope_name = self._graph.unique_name(name, mark_as_used=False)
    relative_scope_name = absolute_scope_name.split("/")[-1]

    import_collections = [
        # In most cases ASSET_FILEPATHS are only used for the TABLE_INITIALIZERS
        # ops, however one could create a graph that uses an asset at any other
        # time. As so everytime we bring the tensor with that has the asset
        # filename we must annotate it as so, so later re-exports have that
        # semantic information and can handle it.
        tf.GraphKeys.ASSET_FILEPATHS,
        tf.GraphKeys.COND_CONTEXT,
        tf.GraphKeys.WHILE_CONTEXT,
    ]
    if self._trainable:
      import_collections.extend([tf.GraphKeys.UPDATE_OPS])

    tf.train.import_meta_graph(
        adapted_meta_graph_for_import(self._meta_graph, absolute_scope_name),
        input_map=feed_map,
        import_scope=relative_scope_name,
        restore_collections_predicate=(lambda key: key in import_collections))
    fix_colocation_after_import(input_map=feed_map,
                                absolute_import_scope=absolute_scope_name)

    def get_tensor(name):
      # When trying to output an input tensor there are no nodes created within
      # the apply scope. So one must look into the input map.
      try:
        return feed_map[name]
      except KeyError:
        return self._graph.get_tensor_by_name(
            prepend_name_scope(name, import_scope=absolute_scope_name))

    return tensor_info.build_output_map(signature_def.outputs, get_tensor)

  def export(self, path, session):
    """See `Module.export`."""
    def variables_saver(variables_path):
      if self._saver:
        self._saver.save(
            session, variables_path,
            write_meta_graph=False,
            write_state=False)

    self._spec._export(path, variables_saver)  # pylint: disable=protected-access

  @property
  def variable_map(self):
    """See `Module.variable_map`."""
    return self._variable_map


def adapted_meta_graph_for_import(meta_graph, absolute_import_scope):
  """Prefixes the shared_name attributes of nodes with the import scope."""
  # The MetaGraphDef is copied (and not modified in-place) so that future
  # imports won't see these modifications.
  copy = meta_graph_pb2.MetaGraphDef()
  copy.CopyFrom(meta_graph)
  shared_name_attr = "shared_name"
  for node in copy.graph_def.node:
    shared_name_value = node.attr.get(shared_name_attr, None)
    if shared_name_value and shared_name_value.HasField("s"):
      if shared_name_value.s:
        node.attr[shared_name_attr].s = tf.compat.as_bytes(
            prepend_name_scope(
                shared_name_value.s, import_scope=absolute_import_scope))
  return copy


def prepend_name_scope(name, import_scope):
  """Prepends name scope to a name."""
  # Based on tensorflow/python/framework/ops.py implementation.
  if import_scope:
    try:
      str_to_replace = r"([\^]|loc:@|^)(.*)"
      return re.sub(str_to_replace, r"\1" + import_scope + r"/\2",
                    tf.compat.as_str(name))
    except TypeError as e:
      # If the name is not of a type we can process, simply return it.
      tf.logging.warning(e)
      return name
  else:
    return name


def list_registered_stateful_ops_without_inputs():
  """Returns set of registered stateful ops that do not expect inputs.

  This list is used to identify the ops to be included in the state-graph and
  that are subsequently fed into the apply-graphs.

  Returns:
    A set of strings.
  """
  return set([
      name
      for name, op in op_def_registry.get_registered_ops().items()
      if op.is_stateful and not op.input_arg
  ])


def get_state_map(meta_graph, state_ops, unsupported_state_ops,
                  get_tensor_by_name):
  """Returns a map from tensor names to tensors that hold the state."""
  state_map = {}
  for node in meta_graph.graph_def.node:
    if node.op in state_ops:
      tensor_name = node.name + ":0"
      state_map[tensor_name] = get_tensor_by_name(tensor_name)
    if node.op in unsupported_state_ops:
      raise ValueError("Unsupported stateful op: %s" % node.op)
  return state_map


def get_node_map_from_tensor_map(tensor_map):
  """Converts the keys from tensor name to node name.

  Args:
    tensor_map: Map where keys are full tensor names and values are tensors.

  Returns:
    Map same as tensor_map, except keys have the output_number stripped.
  """
  return {
      _get_node_name_from_tensor(key): value
      for key, value in tensor_map.items()
  }


def _get_node_name_from_tensor(tensor_name):
  """Given a tensor name in format node_name:output_number returns node_name."""
  result = re.match(r"(.*):\d+$", tensor_name)
  if not result:
    raise ValueError(
        "Unexpected format for tensor name. Expected node_name:output_number. "
        "Got %r" % tensor_name)
  return result.group(1)


def _extract_variable_parts(variable_key, variable):
  """Matches a variable to individual parts.

  Args:
    variable_key: String identifier of the variable in the module scope.
    variable: Variable tensor.

  Returns:
    partitioned: Whether the variable is partitioned.
    name: Name of the variable up to the partitioning.
    offset: Offset of the variable into the full variable.

  Raises:
    RuntimeError: In case of unexpected variable format.
  """
  name, offset, partitioned = None, None, False
  # pylint: disable=protected-access
  if variable._save_slice_info:
    name = variable_key[:variable_key.rfind("/")]
    if not variable._save_slice_info.full_name.endswith(name):
      raise RuntimeError("Unexpected handling of partitioned variable.")
    offset = variable._save_slice_info.var_offset[0]
    partitioned = True
  # pylint: enable=protected-access
  return partitioned, name, offset


def recover_partitioned_variable_map(var_node_map):
  """Builds a proper variable map if it contains PartitionedVariables.

  Args:
    var_node_map: A map to tf.Variables. PartitionedVariables show up in this
      map as N entries with keys "<var_name>/part_n".

  Returns:
    A map to tf.Variables or to list of tf.Variables for each
    PartitionedVariables in `var_node_map`.

  Raises:
    RuntimeError: if there are issues recovering the PartitionedVariables.
  """
  offset_variables_map = {}
  for var_key, var_tensor in var_node_map.items():
    match, var_name, offset = _extract_variable_parts(var_key, var_tensor)

    if not match:
      # This is a standard variable, so we can safely add it to the output.
      if var_key in offset_variables_map:
        raise RuntimeError(
            "Variable %s exists both as a single and partitioned variable.")
      offset_variables_map[var_key] = var_tensor
      continue

    if var_name not in offset_variables_map:
      offset_variables_map[var_name] = {}
    elif not isinstance(offset_variables_map[var_name], dict):
      raise RuntimeError(
          "Variable %s exists both as a single and partitioned variable.")

    # Duplicated variable offsets should not exist.
    if offset in offset_variables_map[var_name]:
      raise RuntimeError(
          "Variable map contains duplicate offset %d for variable [%s]" %
          (offset, var_name))
    offset_variables_map[var_name][offset] = var_tensor

  variables_map = {}
  # Use offsets for sorting, then strip them from the dictionary and keep only
  # a list of variables per each variable name.
  for var_name, var_value in offset_variables_map.items():
    if not isinstance(var_value, dict):
      variables_map[var_name] = var_value
      continue
    shapes = [var_tensor.shape[1:] for var_tensor in var_value.values()]
    if not all(shape == shapes[0] for shape in shapes):
      raise RuntimeError("Shapes not compatible: %s" % (shapes))
    for _, tensor in sorted(var_value.items()):
      variables_map[var_name] = [
          tensor for _, tensor in sorted(var_value.items())
      ]

  return variables_map


def check_unique_tags(tag_list):
  """Checks that tag list contains each set of tags only once."""
  frozen_tags_seen = set()
  for tags in tag_list:
    frozen_tags = frozenset(tags)
    if frozen_tags in frozen_tags_seen:
      raise ValueError("Tags %r used repeatedly" % tags)
    frozen_tags_seen.add(frozen_tags)


def check_collections_are_supported(saved_model_handler, supported):
  """Checks that SavedModelHandler only uses supported collections."""
  for meta_graph in saved_model_handler.meta_graphs:
    used_collection_keys = set(meta_graph.collection_def.keys())
    unsupported = used_collection_keys - supported
    if unsupported:
      raise ValueError("Unsupported collections in graph: %s\n"
                       "Use hub.create_module_spec(..., drop_collections=[...])"
                       " as appropriate." % list(unsupported))


def register_ops_if_needed(graph_ops):
  """Register graph ops absent in op_def_registry, if present in c++ registry.

  Args:
    graph_ops: set with graph op names to register.

  Raises:
    RuntimeError: if `graph_ops` contains ops that are not in either python or
      c++ registry.
  """
  missing_ops = graph_ops - set(op_def_registry.get_registered_ops().keys())

  if not missing_ops:
    return

  p_buffer = c_api.TF_GetAllOpList()
  cpp_op_list = op_def_pb2.OpList()
  cpp_op_list.ParseFromString(c_api.TF_GetBuffer(p_buffer))
  cpp_registry_ops = {op.name: op for op in cpp_op_list.op}

  missing_op_list = op_def_pb2.OpList()
  for missing_op in missing_ops:
    if missing_op not in cpp_registry_ops:
      tf.logging.info(
          "Op %s is missing from both the python and C++ registry.",
          missing_op)
    else:
      missing_op_list.op.extend([cpp_registry_ops[missing_op]])
      tf.logging.info(
          "Adding op %s from c++ registry to python registry.",
          missing_op)

  op_def_registry.register_op_list(missing_op_list)

  # Note: Only raise missing op ValueError after trying to load ops.
  # This allows the test to exercise all the calls into TensorFlow
  # without having to write a C + python test.
  if not missing_ops <= set(cpp_registry_ops.keys()):
    raise RuntimeError(
        "Graph ops missing from the python registry (%s) are also absent from "
        "the c++ registry."
        % missing_ops.difference(set(cpp_registry_ops.keys())))


def fix_colocation_after_import(input_map, absolute_import_scope):
  """Fixes colocation attributes after import according to input_map.

  This function is meant to be called after importing a GraphDef, in order
  to rewrite colocate_with constrains analogous to how inputs to ops
  are rewritten by input_map during import. It also updates devices accordingly.

  The nodes in the given import scope of the current default graph have their
  colocation attributes (that is, the "loc:@..." values in the "_class" attr)
  rewritten as follows: If, before the call, op x has attribute loc:@y, and
  `input_map` replaces an output of y with an output of z, then loc:@y gets
  replaced by the colocation attributes of z (that is, loc:@z, if no other
  constraints are in play).

  This style of rewriting requires that the other nodes in the graph express
  their colocation with state and input nodes solely in terms of the state
  and input nodes themselves, which is what the `input_map` provides. The
  reverse direction (e.g., a state node referencing a non-state node in a
  colocation attribute) is prevented by `find_state_op_colocation_error()` and
  `find_signature_input_colocation_error()`.

  Args:
    input_map: a dict mapping from tensor names in the imported graph to
      existing Tensors, typically the same as passed to tf.import_graph_def().
    absolute_import_scope: a string with the full name of the import scope,
      comprising the current scope when import_graph_def() as called plus
      the import_scope passed to it.

  Raises:
    ValueError: if one imported op has its multiple outputs replaced by
      different existing ops. (This is unexpected, since placeholders and
      the current state ops all have only one output.)
  """
  attr_map = _build_colocation_attr_map(input_map, absolute_import_scope)
  _apply_colocation_attr_map(attr_map, absolute_import_scope)


def _build_colocation_attr_map(input_map, absolute_import_scope):
  """Returns a dict mapping from pre-import to post-import colocation attrs.

  Args:
    input_map: as for fix_colocation_after_import.
    absolute_import_scope: as for fix_colocation_after_import.

  Returns:
    A dict that maps bytes `"loc:@" + absolute_import_scope + "/foo"`
    to lists of bytes `["loc:@...", ...]` that are the colocation_groups
    of the op that replaces the outputs of `foo` according to the `import_map`.

  Raises:
    ValueError: if `input_map` has multiple outputs of one op, and they
      get mapped to existing ops with different colocation groups.
  """
  colocation_attr_map = {}
  for imported_tensor_name, mapped_tensor in input_map.items():
    imported_tensor_name = absolute_import_scope + "/" + imported_tensor_name
    imported_op_name = _get_node_name_from_tensor(imported_tensor_name)
    key = tf.compat.as_bytes("loc:@" + imported_op_name)
    mapped_coloc_groups = mapped_tensor.op.colocation_groups()
    previous_mapped_coloc_groups = colocation_attr_map.get(key)
    if (previous_mapped_coloc_groups and
        set(previous_mapped_coloc_groups) != set(mapped_coloc_groups)):
      raise ValueError("Imported op %s has its outputs mapped to existing ops "
                       "with different colocation_groups: %s vs %s" %
                       (imported_op_name, previous_mapped_coloc_groups,
                        mapped_coloc_groups))
    colocation_attr_map[key] = mapped_coloc_groups
  return colocation_attr_map


def _apply_colocation_attr_map(colocation_attr_map, absolute_import_scope):
  """Rewrites colocation constraints in the current default graph.

  Nodes in `absolute_import_scope` get their "_class" attr lists rewritten
  according to `colocation_attr_map`: each entry that matches a key gets
  replaced by the associated values (with deduplication). The node's device
  is updated accordingly.

  Args:
    colocation_attr_map: as returned by _build_colocation_attr_map.
    absolute_import_scope: as for fix_colocation_after_import.
  """
  graph = tf.get_default_graph()
  for op in graph.get_operations():
    # Rewrite the values of the "_class" attr that store colocation constraints.
    if not op.name.startswith(absolute_import_scope + "/"): continue
    try:
      class_values = op.get_attr("_class")
    except ValueError:
      continue  # No _class attr found; nothing to do.
    new_attr_value = tf.AttrValue()
    new_coloc_groups = []
    for class_value in class_values:
      if class_value.startswith(tf.compat.as_bytes("loc:@")):
        if class_value in colocation_attr_map:
          new_coloc_groups.extend(colocation_attr_map[class_value])
        else:
          new_coloc_groups.append(class_value)
      else:
        new_attr_value.list.s.append(class_value)
    new_coloc_groups = sorted(set(new_coloc_groups))
    new_attr_value.list.s.extend(new_coloc_groups)
    op._set_attr("_class", new_attr_value)  # pylint: disable=protected-access

    # Mimic the code of tf.import_graph_def(): If there are colocation
    # constraints, use any of them to set the device (overriding what the
    # device function stack would do), without attempting to merge or check for
    # equality. If they were inconsistent, TensorFlow's C++ runtime would fail
    # anyways due to conflicting colocation constraints.
    # Note that Hub imports GraphDefs with devices cleared, so this code deals
    # with the result of import_graph_def, not a setting saved in the module.
    if new_coloc_groups:
      new_coloc_device = ""
      for new_coloc_group in new_coloc_groups:
        assert new_coloc_group.startswith(tf.compat.as_bytes("loc:@"))
        new_coloc_target_op = graph.get_operation_by_name(
            tf.compat.as_str(new_coloc_group[5:]))
        new_coloc_device = new_coloc_target_op.device
        if new_coloc_device: break
      # Set this, even if empty, to avoid retaining an outdated value.
      op._set_device(new_coloc_device)  # pylint: disable=protected-access


def find_state_op_colocation_error(graph, reported_tags=None):
  """Returns error message for colocation of state ops, or None if ok."""
  state_op_types = list_registered_stateful_ops_without_inputs()
  state_op_map = {op.name: op for op in graph.get_operations()
                  if op.type in state_op_types}
  for op in state_op_map.values():
    for colocation_group in op.colocation_groups():
      if not (colocation_group.startswith(tf.compat.as_bytes("loc:@")) and
              tf.compat.as_str(colocation_group[5:]) in state_op_map):
        tags_prefix = ("" if reported_tags is None else
                       "in the graph for tags %s, " % reported_tags)
        return (
            "A state-holding node x of a module's graph (e.g., a Variable op) "
            "must not be subject to a tf.colocate_with(y) constraint "
            "unless y is also a state-holding node.\n"
            "Details: %snode '%s' has op '%s', which counts as state-holding, "
            "but Operation.colocation_groups() == %s. " %
            (tags_prefix, op.name, op.type, op.colocation_groups()))
  return None


def find_signature_input_colocation_error(signature_name, inputs):
  """Returns error message for colocation of signature inputs, or None if ok."""
  for input_name, tensor in inputs.items():
    expected_colocation_groups = [tf.compat.as_bytes("loc:@" + tensor.op.name)]
    if tensor.op.colocation_groups() != expected_colocation_groups:
      return (
          "A tensor x used as input in a signature must not be subject to a "
          "tf.colocate_with(y) constraint. (The reverse would be allowed.)\n"
          "Details: tensor '%s' appears as input '%s' of signature '%s' "
          "but has Tensor.op.colocation_groups() == %s" %
          (tensor, input_name, signature_name, tensor.op.colocation_groups()))
  return None
