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
"""The deprecated hub.Module class of TensorFlow Hub."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

import six
import tensorflow as tf

from tensorflow_hub import module_spec
from tensorflow_hub import registry
from tensorflow_hub import tensor_info
from tensorflow_hub import tf_v1


def as_module_spec(spec):
  if isinstance(spec, module_spec.ModuleSpec):
    return spec
  elif isinstance(spec, six.string_types):
    return load_module_spec(spec)
  else:
    raise ValueError("Unknown module spec type: %r" % type(spec))


def load_module_spec(path):
  """Loads a ModuleSpec from a TF Hub service or the filesystem.

  DEPRECATION NOTE: This belongs to the hub.Module API and TF1 Hub format.
  For TF2, switch to plain SavedModels and hub.load(); see also hub.resolve().

  Args:
    path: string describing the location of a module. There are several
      supported path encoding schemes:
        a) A URL like "https://tfhub.dev/the/module/1" referring to tfhub.dev or
           another service implementing https://www.tensorflow.org/hub/hosting.
        b) A URL like "https://example.com/module.tar.gz" that points to a
           compressed tarball directly, as long as that web server ignores
           the query parameters added by https://www.tensorflow.org/hub/hosting.
        c) Any filesystem location of a module directory (e.g. /module_dir
           for a local filesystem). All filesystems implementations provided
           by Tensorflow are supported.
        d) Private name resolution schemes added by the maintainer of your
           local installation of the tensorflow_hub library (usually none).

  Returns:
    A ModuleSpec.

  Raises:
    ValueError: on unexpected values in the module spec.
    tf.errors.OpError: on file handling exceptions.
  """
  path = registry.resolver(path)
  return registry.loader(path)


def export_module_spec(spec, path, checkpoint_path, name_transform_fn):
  """Helper function to ModuleSpec.export()."""
  with tf.Graph().as_default():
    m = Module(spec)
    assign_map = {
        name_transform_fn(name): value for name, value in m.variable_map.items()
    }
    tf_v1.train.init_from_checkpoint(checkpoint_path, assign_map)
    init_op = tf_v1.initializers.global_variables()
    with tf_v1.Session() as session:
      session.run(init_op)
      m.export(path, session)


# Module class provides a unified access to all ModuleSpecs implementations and
# should not contain specific implementation code in it (e.g. SavedModel code).
class Module(object):
  """Part of a TensorFlow 1 model that can be transferred between models.

  DEPRECATION NOTE: The hub.Module API works for TF1 only.
  For TF2, switch to plain SavedModels and hub.load().

  A Module represents a part of a TensorFlow graph that can be exported to disk
  (based on the SavedModel format) and later re-loaded. A Module has a defined
  interface that allows it to be used in a replaceable way, with little or no
  knowledge of its internals and its serialization format. Example:

  ```python
  m = hub.Module("/tmp/text-embedding")
  embeddings = m(sentences)
  ```

  The module to instantiate is defined by its spec (a `ModuleSpec` or a
  path where to load it from) which contains the module weights, assets and
  signatures.

  During instantiation the Module adds the state (e.g. variables and tables ops)
  to the current graph. Afterwards, the method `__call__()` allows to apply the
  module `signatures` multiple times, which adds ops for the computation.

  A Module may provide different variants of its graph for different purposes
  (say, training or serving, which may behave differently, e.g., for batch
  normalization). Graph variants are identified by sets of string-valued tags.
  The graph variant used to create a module that is exported must define all the
  variables needed by any other graph variant that is subsequently used.

  To make it possible to easily replace a module with another, they all assume
  that they will be used with common TensorFlow conventions such as session
  initialization and restore, use of collections for variables, regularization
  losses and updates, etc.
  """

  def __init__(self, spec, trainable=False, name="module", tags=None):
    """Constructs a Module to be used in the current graph.

    This creates the module `state-graph` under an unused variable_scope based
    on `name`. During this call a Module will:

    - Add GLOBAL_VARIABLES under its scope. Those variables may be added to
      to the TRAINABLE_VARIABLES collection (depending on `trainable` parameter)
      and to the MODEL_VARIABLES. The variables must be initialized before use,
      and can be checkpointed as usual.

    - Add ops to the INIT_TABLE_OPS collection, which must be run during session
      initialization and add constant tensors to ASSET_FILEPATHS that are
      needed during the execution of such ops.

    - Add tensors to the REGULARIZATION_LOSSES collection (depending on
      `trainable` parameter).

    Args:
      spec: A ModuleSpec defining the Module to instantiate or a path where
        to load a ModuleSpec from via `load_module_spec`.
      trainable: whether the Module is trainable. If False, no variables are
        added to TRAINABLE_VARIABLES collection, and no tensors are added to
        REGULARIZATION_LOSSES collection.
      name: A string, the variable scope name under which to create the Module.
        It will be uniquified and the equivalent name scope must be unused.
      tags: A set of strings specifying the graph variant to use.

    Raises:
      RuntimeError: explaning the reason why it failed to instantiate the
        Module.
      ValueError: if the requested graph variant does not exists.
      tf.errors.NotFoundError: if the requested graph contains unknown ops.
    """
    self._graph = tf_v1.get_default_graph()
    self._spec = as_module_spec(spec)
    self._trainable = trainable

    self._tags = set(tags or [])
    if self._tags not in self._spec.get_tags():
      tags = sorted(list(tags)) if tags else tags
      raise ValueError("No such graph variant: tags=%r" % tags)

    abs_state_scope = _try_get_state_scope(name, mark_name_scope_used=False)
    self._name = abs_state_scope.split("/")[-2]

    abs_parent_scope = abs_state_scope.split("/")[:-2]
    if abs_parent_scope:
      abs_parent_scope = "/".join(abs_parent_scope) + "/"
    else:
      abs_parent_scope = ""

    with tf.name_scope(abs_parent_scope):
      # pylint: disable=protected-access
      self._impl = self._spec._create_impl(
          name=self._name,
          trainable=self._trainable,
          tags=self._tags)
      # pylint: enable=protected-access

  def __call__(self, inputs=None,  # pylint: disable=invalid-name
               _sentinel=None, signature=None, as_dict=None):
    """Instantiates a module signature in the graph.

    Example calls:

    ```python
      # Use default signature with one input and default output.
      embeddings = m(["hello world", "good morning"])

      # Use "encode" signature with one input and default output.
      encodings = m(["hello world"], signature="encode")

      # Use default signature with input dict and output dict.
      dict_outputs = m({"text": [...], "lang": [...]}, as_dict=True)
    ```

    The method __call__() allows to create the graph ops that compute a
    signature outputs given the inputs and using this module instance state.
    Each signature can be applied multiple times with different inputs and they
    all share the same module state.

    A Module may define multiple signatures. Use `signature=<name>` to identify
    the specific signature to instantiate. If omitted or None, the default
    signature is used.

    A signature may define various outputs. Use `as_dict=True` to return a dict
    of all outputs. If omitted or False, the output named 'default' is
    returned.

    During this call a Module will:

    - Add ops in the current name scope to convert the inputs in tensors to feed
      to the signature.

    - Add ops to the UPDATE_OPS collection which depend on at least one of the
      provided inputs if the Module was constructed with `trainable=True`.

    - Add constant tensors to ASSET_FILEPATHS, even if those are not needed
      directly needed for the signature.

    Note: `hub.Module` implementation depends on graph pruning that happens
    usually during `session.run` as so it can lead to errors when used inside
    function graphs that execute all its ops (e.g. `tf.data.Dataset.map`).

    Args:
      inputs: Inputs to the signature. A dict from input names to tensor
        values. If the signature only expects one input, one may pass
        a single value. If the signature has no inputs, it may be omitted.
      _sentinel: Used to prevent positional parameters besides `inputs`.
      signature: A string with the signature name to apply. If none, the
        default signature is used.
      as_dict: A boolean indicating whether to the return all the outputs
        of the signature as a dict or return only the default output.

    Returns:
      A tensor (single or sparse) if the signature defines a default output or
      a dict from strings (output names) to tensors if `as_dict=True` is used.

    Raises:
      TypeError: If there is a mismatch on arguments, inputs or outputs of
        the module signature.
      RuntimeError: If there are errors during creation of the signature graph.
    """
    if self._graph is not tf_v1.get_default_graph():
      raise RuntimeError(
          "Module must be applied in the graph it was instantiated for.")

    signature = self._impl.get_signature_name(signature)
    # SavedModel non-default signatures automatically includes ':' in them,
    # but that is an invalid character for a name that is used as part
    # of variable scopes.
    safe_signature = signature.replace(":", "_")
    name = "%s_apply_%s" % (self._name, safe_signature)

    dict_inputs = _convert_dict_inputs(
        inputs, self._spec.get_input_info_dict(signature=signature,
                                               tags=self._tags))

    dict_outputs = self._impl.create_apply_graph(
        signature=signature,
        input_tensors=dict_inputs,
        name=name)
    return _prepare_outputs(dict_outputs, as_dict=as_dict)

  def get_signature_names(self):
    """Returns the module's signature names as an iterable of strings."""
    return self._spec.get_signature_names(tags=self._tags)

  def get_input_info_dict(self, signature=None):
    """Describes the inputs required by a signature.

    Args:
      signature: A string with the signature to get inputs information for.
        If None, the default signature is used if defined.

    Returns:
      The result of ModuleSpec.get_input_info_dict() for the given signature,
      and the graph variant selected by `tags` when this Module was initialized.

    Raises:
      KeyError: if there is no such signature.
    """
    return self._spec.get_input_info_dict(signature=signature, tags=self._tags)

  def get_output_info_dict(self, signature=None):
    """Describes the outputs provided by a signature.

    Args:
      signature: A string with the signature to get ouputs information for.
        If None, the default signature is used if defined.

    Returns:
      The result of ModuleSpec.get_output_info_dict() for the given signature,
      and the graph variant selected by `tags` when this Module was initialized.

    Raises:
      KeyError: if there is no such signature.
    """
    return self._spec.get_output_info_dict(signature=signature, tags=self._tags)

  def get_attached_message(self, key, message_type, required=False):
    """Calls ModuleSpec.get_attached_message(); see there for more."""
    return self._spec.get_attached_message(key, message_type,
                                           tags=self._tags, required=required)

  def export(self, path, session):
    """Exports the module with the variables from the session in `path`.

    Note that it is the module definition in the ModuleSpec used to create this
    module that gets exported. The session is only used to provide the value
    of variables.

    Args:
      path: path where to export the module to.
      session: session where to export the variables from.

    Raises:
      RuntimeError: if there is an issue during the export.
    """
    if self._graph is not tf_v1.get_default_graph():
      raise RuntimeError("default graph differs from the graph where the "
                         "module was instantiated.")
    if self._graph is not session.graph:
      raise RuntimeError("session graph differs from the graph where the "
                         "module was instantiated.")
    self._impl.export(path, session)

  @property
  def variable_map(self):
    """Map from original variable names into tf.Variables (or lists of them).

    This map translates between variable names relative to the module and the
    corresponding Variable objects that have been created by instantiating it
    in the current graph (with the applicable scoping added). Each key in the
    map is a variable name as created by running the module's defining
    `module_fn` in the root scope of an empty graph. Each value in the map is
    a Variable object, or in case of partitioned variables a list of Variable
    objects.

    This property can be used with `tf.init_from_checkpoint` as `assignment_map`
    in order to restore a pre-trained checkpoint into a Module before calling
    `Module.export()`.

    Returns:
      A dict from the variable names in the Module to the instantiated
      tf.Variables or list of tf.Variables (if partitioned). The keys of this
      map are the same regardless of the scope of where the Module was
      instantiated.
    """
    return self._impl.variable_map

  @property
  def variables(self):
    """Returns the list of all tf.Variables created by module instantiation."""
    result = []
    for _, value in sorted(self.variable_map.items()):
      if isinstance(value, list):
        result.extend(value)
      else:
        result.append(value)
    return result


def _try_get_state_scope(name, mark_name_scope_used=True):
  """Returns a fresh variable/name scope for a module's state.

  In order to import a module into a given scope without major complications
  we require the scope to be empty. This function deals with deciding an unused
  scope where to define the module state. This is non trivial in cases where
  name_scope and variable_scopes are out of sync, e.g. tpus or re-entering
  scopes.

  Args:
    name: A string with the name of the module as supplied by the client.
    mark_name_scope_used: a boolean, indicating whether to mark the name
        scope of the returned value as used.

  Raises:
    RuntimeError: if the name scope of the freshly created variable scope is
        already used.
  """
  tmp_scope_name = tf_v1.get_variable_scope().name
  if tmp_scope_name:
    tmp_scope_name += "/"
  with tf.name_scope(tmp_scope_name):
    # Pick an unused variable scope.
    with tf_v1.variable_scope(
        None, default_name=name, auxiliary_name_scope=False) as vs:
      abs_state_scope = vs.name + "/"
    # Verify that the name scope is available and mark it used if requested.
    graph = tf_v1.get_default_graph()
    unique_name_scope = graph.unique_name(name, mark_name_scope_used) + "/"
    if unique_name_scope != abs_state_scope:
      raise RuntimeError(
          "variable_scope %s was unused but the corresponding "
          "name_scope was already taken." % abs_state_scope)
  return abs_state_scope


def _prepare_dict_inputs(inputs, tensor_info_map):
  """Converts inputs to a dict of inputs and checks extra/missing args.

  Args:
    inputs: inputs fed to Module.__call__().
    tensor_info_map: A map from string to `tensor_info.ParsedTensorInfo`
      describing the signature inputs.

  Returns:
    A dict of values with the same keys as tensor_info_map.

  Raises:
    TypeError: If it fails to convert the input values into a dict of tensors
      to feed to the signature instantiation.
  """
  if inputs is None:
    dict_inputs = {}
  elif isinstance(inputs, dict):
    dict_inputs = inputs
  elif len(tensor_info_map) == 1:
    dict_inputs = {list(tensor_info_map.keys())[0]: inputs}
  elif not tensor_info_map:
    raise TypeError("Signature expects no inputs.")
  else:
    raise TypeError("Signature expects multiple inputs. Use a dict.")

  dict_inputs_keys = set(dict_inputs.keys())
  tensor_info_map_keys = set(tensor_info_map.keys())
  if dict_inputs_keys != tensor_info_map_keys:
    raise TypeError("Cannot convert dict_inputs: missing %r, extra given %r" %
                    (sorted(list(tensor_info_map_keys - dict_inputs_keys)),
                     sorted(list(dict_inputs_keys - tensor_info_map_keys))))

  return dict_inputs


def _convert_dict_inputs(inputs, tensor_info_map):
  """Converts from inputs into dict of input tensors.

  This handles:
    - putting inputs into a dict, per _prepare_dict_inputs(),
    - converting all input values into tensors compatible with the
      expected input tensor (dtype, shape).
    - check sparse/non-sparse tensor types.

  Args:
    inputs: inputs fed to Module.__call__().
    tensor_info_map: A map from string to `tensor_info.ParsedTensorInfo`
      describing the signature inputs.

  Returns:
    A dict of tensors to feed to the signature instantiation.

  Raises:
    TypeError: If it fails to convert the input values into a dict of tensors
      to feed to the signature instantiation.
  """
  dict_inputs = _prepare_dict_inputs(inputs, tensor_info_map)
  return tensor_info.convert_dict_to_compatible_tensor(dict_inputs,
                                                       tensor_info_map)


def _prepare_outputs(dict_outputs, as_dict):
  """Converts from dict outputs into the return value of Module.__call__().

  Args:
    dict_outputs: A dict output from applying a signature.
    as_dict: A boolean indicating whether to return the outputs of the Module
      as a dict or return the output named 'default.

  Returns:
    A tensor with the output named 'default' or a dict of output tensors if
    `as_dict=True`.

  Raises:
    TypeError: If as_dict is False and there is no output named 'default'.
  """
  if as_dict:
    return dict_outputs
  if "default" in dict_outputs:
    return dict_outputs["default"]
  else:
    raise TypeError("There is no output named 'default'. Use as_dict=True.")


@contextlib.contextmanager
def eval_function_for_module(spec, tags=None):
  """Context manager that yields a function to directly evaluate a hub.Module.

  DEPRECATION NOTE: This belongs to the hub.Module API and TF1 Hub format.
  For TF2, switch to plain SavedModels and hub.load().
  Eager evalutaion in TF2 obviates the need for this helper.

  This creates a separate graph, in which all of the signatures of the module
  are instantiated. Then, it creates a session and initializes the module
  variables. Finally, it returns a function which can be used to evaluate the
  module signatures.

  The function returned by eval_function_for_module has the same syntax as
  Module.__call__ , except that inputs and outputs are not tensors but actual
  values as used with Session.run().

  ```python
  with hub.eval_function_for_module("/tmp/text-embedding") as f:
    # The module can be directly evaluated using f without constructing a graph.
    embeddings = f(["Hello world!",], signature="mysignature")
  ```

  Args:
    spec: A ModuleSpec defining the Module to instantiate or a path where to
      load a ModuleSpec from via `load_module_spec`.
    tags: A set of strings specifying the graph variant to use.

  Yields:
    A function whose keyword arguments are fed into the tfhub module and which
      returns a dictionary with the value of the output tensors.

  Raises:
    RuntimeError: explaning the reason why it failed to instantiate the
      Module.
    ValueError: if the requested graph variant does not exists.
  """
  # We create a separate graph and add all the signatures of the module to it.
  original_graph = tf_v1.get_default_graph()
  with tf.Graph().as_default():
    module = Module(spec, tags=tags)
    input_tensors_per_signature = {}
    output_tensors_per_signature = {}
    for signature in module.get_signature_names():
      # We scope with the signature name as different signatures will likely
      # contain tensors with the same name (e.g. the input and output tensors).
      with tf_v1.variable_scope(signature):
        input_tensors = {}
        for name, tensorinfo in module.get_input_info_dict(signature).items():
          # We need to be care with the shape as it may be fully-known,
          # partially-known or even unknown.
          shape = tensorinfo.get_shape()
          effective_shape = None if shape.dims is None else shape.as_list()
          if tensorinfo.is_sparse:
            input_tensors[name] = tf_v1.sparse_placeholder(
                tensorinfo.dtype, shape=effective_shape, name=name)
          else:
            input_tensors[name] = tf_v1.placeholder(
                tensorinfo.dtype, shape=effective_shape, name=name)
        input_tensors_per_signature[signature] = input_tensors
        output_tensors_per_signature[signature] = module(
            input_tensors_per_signature[signature],
            signature=signature,
            as_dict=True)

  # Evaluating the tfhub module requires an active tensorflow session.
    with tf_v1.train.SingularMonitoredSession() as sess:

      def func(
          inputs=None,
          _sentinel=None,  # pylint: disable=invalid-name
          signature=None,
          as_dict=None):
        """Function that directly evaluates a signature in the module."""
        signature = signature or "default"
        input_tensors = input_tensors_per_signature[signature]

        dict_inputs = _prepare_dict_inputs(inputs, input_tensors)

        # The input arguments are directly fed into the session.
        feed_dict = {
            input_tensors[key]: value for key, value in dict_inputs.items()
        }
        output = output_tensors_per_signature[signature]
        output = _prepare_outputs(output, as_dict)
        return sess.run(output, feed_dict=feed_dict)

      with original_graph.as_default():
        # Yield the function since that will keep the session alive until the
        # user exits the context.
        yield func
