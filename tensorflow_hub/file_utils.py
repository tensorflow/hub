# Copyright 2020 The TensorFlow Hub Authors. All Rights Reserved.
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
"""Utilities for file operations."""


import os
import tarfile

import tensorflow as tf


def extract_file(tgz,
                 tarinfo,
                 dst_path,
                 buffer_size=10 << 20,
                 log_function=None):
  """Extracts 'tarinfo' from 'tgz' and writes to 'dst_path'."""
  src = tgz.extractfile(tarinfo)
  if src is None:
    return
  dst = tf.compat.v1.gfile.GFile(dst_path, "wb")
  while 1:
    buf = src.read(buffer_size)
    if not buf:
      break
    dst.write(buf)
    if log_function is not None:
      log_function(len(buf))
  dst.close()
  src.close()


def extract_tarfile_to_destination(fileobj, dst_path, log_function=None):
  """Extract a tarfile. Optional: log the progress."""
  with tarfile.open(mode="r|*", fileobj=fileobj) as tgz:
    for tarinfo in tgz:
      abs_target_path = merge_relative_path(dst_path, tarinfo.name)

      if tarinfo.isfile():
        extract_file(tgz, tarinfo, abs_target_path, log_function=log_function)
      elif tarinfo.isdir():
        tf.compat.v1.gfile.MakeDirs(abs_target_path)
      else:
        # We do not support symlinks and other uncommon objects.
        raise ValueError("Unexpected object type in tar archive: %s" %
                         tarinfo.type)


def merge_relative_path(dst_path, rel_path):
  """Merge a relative tar file to a destination (which can be "gs://...")."""
  # Convert rel_path to be relative and normalize it to remove ".", "..", "//",
  # which are valid directories in fileystems like "gs://".
  norm_rel_path = os.path.normpath(rel_path.lstrip("/"))

  if norm_rel_path == ".":
    return dst_path

  # Check that the norm rel path does not starts with "..".
  if norm_rel_path.startswith(".."):
    raise ValueError("Relative path %r is invalid." % rel_path)

  merged = os.path.join(dst_path, norm_rel_path)

  # After merging verify that the merged path keeps the original dst_path.
  if not merged.startswith(dst_path):
    raise ValueError("Relative path %r is invalid. Failed to merge with %r." %
                     (rel_path, dst_path))
  return merged
