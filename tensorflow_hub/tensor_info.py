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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_hub import tf_v1


class ParsedTensorInfo(object):
  """This is a tensor-looking object with information about a Tensor.

  This class provides a subset of methods and attributes provided by real
  instantiated Tensor/SparseTensors in a graph such that code designed to
  handle instances of it would mostly work in real Tensors.
  """

  def __init__(self, dtype, shape, is_sparse):
    self._dtype = dtype
    self._shape = shape
    self._is_sparse = is_sparse

  @property
  def dtype(self):
    """The `DType` of elements in this tensor."""
    return self._dtype

  def get_shape(self):
    """The `TensorShape` that represents the dense shape of this tensor."""
    return self._shape

  @property
  def is_sparse(self):
    """Whether it represents a sparse tensor."""
    # This property is non-standard and does not exist in tf.Tensor or
    # tf.SparseTensor instances.
    return self._is_sparse

  def __repr__(self):
    return "<hub.ParsedTensorInfo shape=%s dtype=%s is_sparse=%s>" % (
        self.get_shape(),
        self.dtype.name,
        self.is_sparse)


def _parse_tensor_info_proto(tensor_info):
  """Returns a ParsedTensorInfo instance from a TensorInfo proto."""
  encoding = tensor_info.WhichOneof("encoding")
  dtype = tf.DType(tensor_info.dtype)
  shape = tf.TensorShape(tensor_info.tensor_shape)
  if encoding == "name":
    return ParsedTensorInfo(dtype=dtype, shape=shape, is_sparse=False)
  elif encoding == "coo_sparse":
    return ParsedTensorInfo(dtype=dtype, shape=shape, is_sparse=True)
  else:
    raise ValueError("Unsupported TensorInfo encoding %r" % encoding)


def parse_tensor_info_map(protomap):
  """Converts a proto map<string, TensorInfo> into a native Python dict.

  The keys are preserved. The TensorInfo protos are parsed into objects
  with dtype property and get_shape() method similar to Tensor and SparseTensor
  objects and an additional `is_sparse` property.

  Args:
    protomap: A proto map<string, TensorInfo>.

  Returns:
    A map from the original keys to python objects.
  """
  return {
      key: _parse_tensor_info_proto(value)
      for key, value in protomap.items()
  }


def _is_sparse(x):
  """Returns whether x is a SparseTensor or a parsed sparse tensor info."""
  return (
      isinstance(x, (tf.SparseTensor, tf_v1.SparseTensorValue)) or
      (hasattr(x, "is_sparse") and x.is_sparse))


def _convert_to_compatible_tensor(value, target, error_prefix):
  """Converts `value` into a tensor that can be feed into `tensor_info`.

  Args:
    value: A value to convert into Tensor or SparseTensor.
    target: An object returned by `parse_tensor_info_map`.
    error_prefix: A string to prefix on raised TypeErrors.

  Raises:
    TypeError: If it fails to convert.

  Returns:
    A Tensor or SparseTensor compatible with tensor_info.
  """
  try:
    tensor = tf_v1.convert_to_tensor_or_indexed_slices(value, target.dtype)
  except TypeError as e:
    raise TypeError("%s: %s" % (error_prefix, e))
  if _is_sparse(tensor) != _is_sparse(target):
    if _is_sparse(tensor):
      raise TypeError("%s: Is sparse. Expected dense." % error_prefix)
    else:
      raise TypeError("%s: Is dense. Expected sparse." % error_prefix)
  if not tensor.get_shape().is_compatible_with(target.get_shape()):
    raise TypeError("%s: Shape %r is incompatible with %r" %
                    (error_prefix, tensor.get_shape(), target.get_shape()))
  return tensor


def convert_dict_to_compatible_tensor(values, targets):
  """Converts dict `values` in tensors that are compatible with `targets`.

  Args:
    values: A dict to objects to convert with same keys as `targets`.
    targets: A dict returned by `parse_tensor_info_map`.

  Returns:
    A map with the same keys as `values` but values converted into
    Tensor/SparseTensors that can be fed into `protomap`.

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
    inputs: A map with same keys as `protomap` of Tensors and SparseTensors.

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
    A map from string to Tensor or SparseTensor instances built from `protomap`
    and resolving tensors using `get_tensor_by_name()`.

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
    else:
      raise ValueError("Invalid TensorInfo.encoding: %s" % encoding)

  return {
      key: get_output_from_tensor_info(tensor_info)
      for key, tensor_info in protomap.items()
  }


def _shape_match(a, b):
  # TRICKY: as_list() can't be used if the number of dimensions is unknown.
  # So we check those before.
  if a.ndims != b.ndims:
    return False
  if a.ndims and a.as_list() != b.as_list():
    return False
  return True


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
    if _is_sparse(info_a[1]) != _is_sparse(info_b[1]):
      return False
    if info_a[1].dtype != info_b[1].dtype:
      return False
    if not _shape_match(info_a[1].get_shape(), info_b[1].get_shape()):
      return False
  return True
