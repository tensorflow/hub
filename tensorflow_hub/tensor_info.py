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
"""TensorFlow Hub internal utilities to handle information about tensors.

This file provides utilities to refer to properties of un-instantiated Tensors
in a concise way. Note: Ideally TensorFlow would provide a way to do this.
"""

import tensorflow as tf

from tensorflow_hub import tf_utils


class ParsedTensorInfo(object):
  """This is a tensor-looking object with information about a Tensor.

  This class provides a subset of methods and attributes provided by real
  instantiated Tensor/SparseTensors/CompositeTensors in a graph such that code
  designed to handle instances of it would mostly work in real Tensors.
  """

  def __init__(self, dtype, shape, is_sparse, type_spec=None):
    if type_spec is not None:
      assert dtype is None and shape is None and is_sparse is None
      self._type_spec = type_spec
    elif is_sparse:
      self._type_spec = tf.SparseTensorSpec(shape, dtype)
    else:
      self._type_spec = tf.TensorSpec(shape, dtype)

  @classmethod
  def from_type_spec(cls, type_spec):
    return cls(None, None, None, type_spec)

  @property
  def dtype(self):
    """The `DType` of elements in this tensor."""
    if hasattr(self._type_spec, "dtype"):
      return self._type_spec.dtype
    elif hasattr(self._type_spec, "_dtype"):
      # Prior to TF version 2.3, RaggedTensor._dtype was private.
      return self._type_spec._dtype  # pylint: disable=protected-access
    else:
      raise ValueError("Expected TypeSpec %r to have a dtype attribute"
                       % self._type_spec)

  def get_shape(self):
    """The `TensorShape` that represents the dense shape of this tensor."""
    if hasattr(self._type_spec, "shape"):
      return self._type_spec.shape
    elif hasattr(self._type_spec, "_shape"):
      # Prior to TF version 2.3, RaggedTensor._shape was private.
      return self._type_spec._shape  # pylint: disable=protected-access
    else:
      raise ValueError("Expected TypeSpec %r to have a shape attribute"
                       % self._type_spec)

  @property
  def is_sparse(self):
    """Whether it represents a sparse tensor."""
    # This property is non-standard and does not exist in tf.Tensor or
    # tf.SparseTensor instances.
    return isinstance(self._type_spec, tf.SparseTensorSpec)

  @property
  def is_composite(self):
    """Whether it represents a composite tensor.  (True for SparseTensor.)"""
    return not isinstance(self._type_spec, tf.TensorSpec)

  @property
  def type_spec(self):
    """`tf.TypeSpec` describing this value's type."""
    return self._type_spec

  @property
  def is_supported_type(self):
    return issubclass(self._type_spec.value_type,
                      tf_utils.SUPPORTED_ARGUMENT_TYPES)

  def __repr__(self):
    if isinstance(self._type_spec, (tf.TensorSpec, tf.SparseTensorSpec)):
      return "<hub.ParsedTensorInfo shape=%s dtype=%s is_sparse=%s>" % (
          self.get_shape(), self.dtype.name, self.is_sparse)
    else:
      return "<hub.ParsedTensorInfo type_spec=%s>" % self.type_spec


def _parse_tensor_info_proto(tensor_info):
  """Returns a ParsedTensorInfo instance from a TensorInfo proto."""
  encoding = tensor_info.WhichOneof("encoding")
  if encoding == "name":
    dtype = tf.DType(tensor_info.dtype)
    shape = tf.TensorShape(tensor_info.tensor_shape)
    return ParsedTensorInfo(dtype=dtype, shape=shape, is_sparse=False)
  elif encoding == "coo_sparse":
    dtype = tf.DType(tensor_info.dtype)
    shape = tf.TensorShape(tensor_info.tensor_shape)
    return ParsedTensorInfo(dtype=dtype, shape=shape, is_sparse=True)
  elif encoding == "composite_tensor":
    spec = tf_utils.composite_tensor_info_to_type_spec(tensor_info)
    return ParsedTensorInfo.from_type_spec(spec)
  else:
    raise ValueError("Unsupported TensorInfo encoding %r" % encoding)


def parse_tensor_info_map(protomap):
  """Converts a proto map<string, TensorInfo> into a native Python dict.

  The keys are preserved. The TensorInfo protos are parsed into objects
  with dtype property and get_shape() method similar to Tensor, SparseTensor,
  and RaggedTensor objects, and additional `is_sparse` and `is_composite`
  properties.

  Args:
    protomap: A proto map<string, TensorInfo>.

  Returns:
    A map from the original keys to python objects.
  """
  return {
      key: _parse_tensor_info_proto(value)
      for key, value in protomap.items()
  }


def _get_type_spec(value):
  if isinstance(value, ParsedTensorInfo):
    return value.type_spec
  elif tf_utils.is_composite_tensor(value):
    return tf_utils.get_composite_tensor_type_spec(value)
  else:
    return tf.TensorSpec.from_tensor(value)


def _convert_to_compatible_tensor(value, target, error_prefix):
  """Converts `value` into a tensor that can be feed into `tensor_info`.

  Args:
    value: A value to convert into Tensor or CompositeTensor.
    target: An object returned by `parse_tensor_info_map`.
    error_prefix: A string to prefix on raised TypeErrors.

  Raises:
    TypeError: If it fails to convert.

  Returns:
    A Tensor or CompositeTensor compatible with tensor_info.
  """
  if tf_utils.is_composite_tensor(value):
    tensor = value
  else:
    try:
      tensor = tf.compat.v1.convert_to_tensor_or_indexed_slices(
          value, target.dtype)
    except TypeError as e:
      raise TypeError("%s: %s" % (error_prefix, e))
  tensor_type_spec = _get_type_spec(tensor)
  target_type_spec = _get_type_spec(target)
  if not ParsedTensorInfo.from_type_spec(tensor_type_spec).is_supported_type:
    raise ValueError(
        "%s: Passed argument of type %s, which is not supported by this "
        "version of tensorflow_hub."
        % (error_prefix, tensor_type_spec.value_type.__name__))

  if not tensor_type_spec.is_compatible_with(target_type_spec):
    if tensor_type_spec.value_type != target_type_spec.value_type:
      got = tensor_type_spec.value_type.__name__
      expected = target_type_spec.value_type.__name__
    else:
      got = str(tensor_type_spec)
      expected = str(target_type_spec)
    raise TypeError("%s: Got %s. Expected %s." % (error_prefix, got, expected))
  return tensor


def convert_dict_to_compatible_tensor(values, targets):
  """Converts dict `values` in tensors that are compatible with `targets`.

  Args:
    values: A dict to objects to convert with same keys as `targets`.
    targets: A dict returned by `parse_tensor_info_map`.

  Returns:
    A map with the same keys as `values` but values converted into
    Tensor/CompositeTensor that can be fed into `protomap`.

  Raises:
    TypeError: If it fails to convert.
  """
  result = {}
  for key, value in sorted(values.items()):
    result[key] = _convert_to_compatible_tensor(
        value, targets[key], error_prefix="Can't convert %r" % key)
  return result


def build_input_map(protomap, inputs):
  """Builds a map to feed tensors in `protomap` using `inputs`.

  Args:
    protomap: A proto map<string,TensorInfo>.
    inputs: A map with same keys as `protomap` of Tensors and CompositeTensors.

  Returns:
    A map from nodes refered by TensorInfo protos to corresponding input
    tensors.

  Raises:
    ValueError: if a TensorInfo proto is malformed or map keys do not match.
  """
  if set(protomap.keys()) != set(inputs.keys()):
    raise ValueError("build_input_map: keys do not match.")
  input_map = {}
  for key, tensor_info in protomap.items():
    arg = inputs[key]
    encoding = tensor_info.WhichOneof("encoding")
    if encoding == "name":
      input_map[tensor_info.name] = arg
    elif encoding == "coo_sparse":
      coo_sparse = tensor_info.coo_sparse
      input_map[coo_sparse.values_tensor_name] = arg.values
      input_map[coo_sparse.indices_tensor_name] = arg.indices
      input_map[coo_sparse.dense_shape_tensor_name] = arg.dense_shape
    elif encoding == "composite_tensor":
      component_infos = tensor_info.composite_tensor.components
      component_tensors = tf.nest.flatten(arg, expand_composites=True)
      for (info, tensor) in zip(component_infos, component_tensors):
        input_map[info.name] = tensor
    else:
      raise ValueError("Invalid TensorInfo.encoding: %s" % encoding)
  return input_map


def build_output_map(protomap, get_tensor_by_name):
  """Builds a map of tensors from `protomap` using `get_tensor_by_name`.

  Args:
    protomap: A proto map<string,TensorInfo>.
    get_tensor_by_name: A lambda that receives a tensor name and returns a
      Tensor instance.

  Returns:
    A map from string to Tensor or CompositeTensor instances built from
    `protomap` and resolving tensors using `get_tensor_by_name()`.

  Raises:
    ValueError: if a TensorInfo proto is malformed.
  """

  def get_output_from_tensor_info(tensor_info):
    encoding = tensor_info.WhichOneof("encoding")
    if encoding == "name":
      return get_tensor_by_name(tensor_info.name)
    elif encoding == "coo_sparse":
      return tf.SparseTensor(
          get_tensor_by_name(tensor_info.coo_sparse.indices_tensor_name),
          get_tensor_by_name(tensor_info.coo_sparse.values_tensor_name),
          get_tensor_by_name(tensor_info.coo_sparse.dense_shape_tensor_name))
    elif encoding == "composite_tensor":
      type_spec = tf_utils.composite_tensor_info_to_type_spec(tensor_info)
      components = [
          get_tensor_by_name(component.name)
          for component in tensor_info.composite_tensor.components
      ]
      return tf_utils.composite_tensor_from_components(type_spec, components)
    else:
      raise ValueError("Invalid TensorInfo.encoding: %s" % encoding)

  return {
      key: get_output_from_tensor_info(tensor_info)
      for key, tensor_info in protomap.items()
  }


def tensor_info_proto_maps_match(map_a, map_b):
  """Whether two signature inputs/outputs match in dtype, shape and sparsity.

  Args:
    map_a: A proto map<string,TensorInfo>.
    map_b: A proto map<string,TensorInfo>.

  Returns:
    A boolean whether `map_a` and `map_b` tensors have the same dtype, shape and
    sparsity.
  """
  iter_a = sorted(parse_tensor_info_map(map_a).items())
  iter_b = sorted(parse_tensor_info_map(map_b).items())
  if len(iter_a) != len(iter_b):
    return False  # Mismatch count.
  for info_a, info_b in zip(iter_a, iter_b):
    if info_a[0] != info_b[0]:
      return False  # Mismatch keys.
    if info_a[1].type_spec != info_b[1].type_spec:
      return False
  return True
