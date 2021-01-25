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
"""Replicates TensorFlow utilities which are not part of the public API."""

import os
import time
import uuid

from absl import logging
import tensorflow as tf


try:
  # pylint: disable=g-direct-tensorflow-import
  # pylint: disable=g-import-not-at-top
  from tensorflow.core.protobuf import struct_pb2
  from tensorflow.python.saved_model import nested_structure_coder
  # pylint: enable=g-direct-tensorflow-import
  # pylint: enable=g-import-not-at-top
except ImportError:
  struct_pb2 = None
  nested_structure_coder = None


def read_file_to_string(filename):
  """Returns the entire contents of a file to a string.

  Args:
    filename: string, path to a file
  """
  return tf.compat.v1.gfile.GFile(filename, mode="r").read()


def atomic_write_string_to_file(filename, contents, overwrite):
  """Writes to `filename` atomically.

  This means that when `filename` appears in the filesystem, it will contain
  all of `contents`. With write_string_to_file, it is possible for the file
  to appear in the filesystem with `contents` only partially written.

  Accomplished by writing to a temp file and then renaming it.

  Args:
    filename: string, pathname for a file
    contents: string, contents that need to be written to the file
    overwrite: boolean, if false it's an error for `filename` to be occupied by
      an existing file.
  """
  temp_pathname = (tf.compat.as_bytes(filename) +
                   tf.compat.as_bytes(".tmp") +
                   tf.compat.as_bytes(uuid.uuid4().hex))
  with tf.compat.v1.gfile.GFile(temp_pathname, mode="w") as f:
    f.write(contents)
  try:
    tf.compat.v1.gfile.Rename(temp_pathname, filename, overwrite)
  except tf.errors.OpError:
    tf.compat.v1.gfile.Remove(temp_pathname)
    raise


# When we create a timestamped directory, there is a small chance that the
# directory already exists because another worker is also writing exports.
# In this case we just wait one second to get a new timestamp and try again.
# If this fails several times in a row, then something is seriously wrong.
MAX_DIRECTORY_CREATION_ATTEMPTS = 10


def get_timestamped_export_dir(export_dir_base):
  """Builds a path to a new subdirectory within the base directory.

  Each export is written into a new subdirectory named using the
  current time.  This guarantees monotonically increasing version
  numbers even across multiple runs of the pipeline.
  The timestamp used is the number of seconds since epoch UTC.

  Args:
    export_dir_base: A string containing a directory to write the exported
        graph and checkpoints.
  Returns:
    The full path of the new subdirectory (which is not actually created yet).

  Raises:
    RuntimeError: if repeated attempts fail to obtain a unique timestamped
      directory name.
  """
  attempts = 0
  while attempts < MAX_DIRECTORY_CREATION_ATTEMPTS:
    export_timestamp = int(time.time())

    export_dir = os.path.join(
        tf.compat.as_bytes(export_dir_base),
        tf.compat.as_bytes(str(export_timestamp)))
    if not tf.compat.v1.gfile.Exists(export_dir):
      # Collisions are still possible (though extremely unlikely): this
      # directory is not actually created yet, but it will be almost
      # instantly on return from this function.
      return export_dir
    time.sleep(1)
    attempts += 1
    logging.warn(
        "Export directory %s already exists; retrying (attempt %d/%d)",
        export_dir, attempts, MAX_DIRECTORY_CREATION_ATTEMPTS)
  raise RuntimeError("Failed to obtain a unique export directory name after "
                     "%d attempts.".MAX_DIRECTORY_CREATION_ATTEMPTS)


def get_temp_export_dir(timestamped_export_dir):
  """Builds a directory name based on the argument but starting with 'temp-'.

  This relies on the fact that TensorFlow Serving ignores subdirectories of
  the base directory that can't be parsed as integers.

  Args:
    timestamped_export_dir: the name of the eventual export directory, e.g.
      /foo/bar/<timestamp>

  Returns:
    A sister directory prefixed with 'temp-', e.g. /foo/bar/temp-<timestamp>.
  """
  dirname, basename = os.path.split(tf.compat.as_bytes(timestamped_export_dir))
  return os.path.join(dirname, b"temp-" + basename)


# Note: This is written from scratch to mimic the pattern in:
# `tf.compat.v1.estimator.LatestExporter._garbage_collect_exports()`.
def garbage_collect_exports(export_dir_base, exports_to_keep):
  """Deletes older exports, retaining only a given number of the most recent.

  Export subdirectories are assumed to be named with monotonically increasing
  integers; the most recent are taken to be those with the largest values.

  Args:
    export_dir_base: the base directory under which each export is in a
      versioned subdirectory.
    exports_to_keep: Number of exports to keep. Older exports will be garbage
      collected. Set to None to disable.
  """
  if exports_to_keep is None:
    return
  version_paths = []  # List of tuples (version, path)
  for filename in tf.compat.v1.gfile.ListDirectory(export_dir_base):
    path = os.path.join(
        tf.compat.as_bytes(export_dir_base),
        tf.compat.as_bytes(filename))
    if len(filename) == 10 and filename.isdigit():
      version_paths.append((int(filename), path))

  oldest_version_path = sorted(version_paths)[:-exports_to_keep]
  for _, path in oldest_version_path:
    try:
      tf.compat.v1.gfile.DeleteRecursively(path)
    except tf.errors.NotFoundError as e:
      logging.warn("Can not delete %s recursively: %s", path, e)


def bytes_to_readable_str(num_bytes, include_b=False):
  """Generate a human-readable string representing number of bytes.

  The units B, kB, MB and GB are used.

  Args:
    num_bytes: (`int` or None) Number of bytes.
    include_b: (`bool`) Include the letter B at the end of the unit.

  Returns:
    (`str`) A string representing the number of bytes in a human-readable way,
      including a unit at the end.
  """

  if num_bytes is None:
    return str(num_bytes)
  if num_bytes < 1024:
    result = "%d" % num_bytes
  elif num_bytes < 1048576:
    result = "%.2fk" % (num_bytes / float(1 << 10))
  elif num_bytes < 1073741824:
    result = "%.2fM" % (num_bytes / float(1 << 20))
  else:
    result = "%.2fG" % (num_bytes / float(1 << 30))

  if include_b:
    result += "B"
  return result


def absolute_path(path):
  """Returns absolute path.

  Args:
    path: Path to compute absolute path from.

  This implementation avoids calling os.path.abspath(path) if 'path' already
  represents an absolute Tensorflow filesystem location (e.g. <fs type>://).
  """
  return path if b"://" in tf.compat.as_bytes(path) else os.path.abspath(path)


# A allowlist of argument types that are supported by hub.Module.  In theory,
# any composite tensor type should work, but since this is a deprecated
# interface, we are limiting support to explicitly tested types.
SUPPORTED_ARGUMENT_TYPES = (tf.Tensor, tf.SparseTensor, tf.RaggedTensor)


# The following helper functions (`is_composite_tensor`,
# `get_composite_tensor_type_spec`, `composite_tensor_info_to_type_spec`, and
# `composite_tensor_from_components`) are used to access composite tensors
# (aka TF Extension Types) in a manner that is both backwards compatible with
# versions of TensorFlow that did not include composite tensors, and forward
# compatible with the TF Extension Types RFC:
# https://github.com/tensorflow/community/blob/eb657a00e8c8c6dbdacbd7d06c304825ba0effd8/rfcs/20200721-extension-types.md


def is_composite_tensor(x):
  """Returns true if `x` is a CompositeTensor."""
  return get_composite_tensor_type_spec(x) is not None


def get_composite_tensor_type_spec(x):
  """Returns the TypeSpec for `x`, or `None` if it's not a composite tensor."""
  type_spec = getattr(x, "__tf_type_spec__", None)
  if type_spec is None:
    return getattr(x, "_type_spec", None)
  else:
    return type_spec()


def composite_tensor_info_to_type_spec(tensor_info):
  """Converts a `TensorInfo` for a composite tensor to a `TypeSpec` object."""
  if nested_structure_coder is None or struct_pb2 is None:
    raise ValueError("This version of TensorFlow does not support "
                     "composite tensors.")
  if tensor_info.WhichOneof("encoding") != "composite_tensor":
    raise ValueError("Expected a TensorInfo with encoding=composite_tensor")
  spec_proto = struct_pb2.StructuredValue(
      type_spec_value=tensor_info.composite_tensor.type_spec)
  struct_coder = nested_structure_coder.StructureCoder()
  return struct_coder.decode_proto(spec_proto)


def composite_tensor_from_components(type_spec, components):
  if hasattr(type_spec, "from_components"):
    return type_spec.from_components(components)
  elif hasattr(type_spec, "_from_components"):
    return type_spec._from_components(components)  # pylint: disable=protected-access
  else:
    raise ValueError(
        "Expected a TypeSpec with a from_components method, got: %r" %
        (type_spec,))
