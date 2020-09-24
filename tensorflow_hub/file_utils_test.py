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
"""Tests for tensorflow_hub.file_utils."""

import os
import tarfile
import tempfile

import tensorflow as tf
from tensorflow_hub import file_utils


class FileUtilsTest(tf.test.TestCase):

  def test_merge_relative_paths(self):
    self.assertEqual(
        file_utils.merge_relative_path("gs://module-cache", ""),
        "gs://module-cache")
    self.assertEqual(
        file_utils.merge_relative_path("gs://module-cache", "./"),
        "gs://module-cache")
    self.assertEqual(
        file_utils.merge_relative_path("gs://module-cache", "./file"),
        "gs://module-cache/file")
    self.assertEqual(
        file_utils.merge_relative_path("gs://module-cache", "hello/../bla"),
        "gs://module-cache/bla")
    self.assertEqual(
        file_utils.merge_relative_path("gs://module-cache", "/"),
        "gs://module-cache", "/")
    with self.assertRaisesRegex(ValueError, "is invalid"):
      file_utils.merge_relative_path("gs://module-cache", "/../")
    with self.assertRaisesRegex(ValueError, "is invalid"):
      file_utils.merge_relative_path("gs://module-cache", "hello/../../bla")

  def test_file_extraction(self):
    # Compress, extract and compare the following directory:
    # /a:               content1
    # /sub_directory/b: content2

    gfile = tf.compat.v1.gfile
    outer_file = "a"
    outer_content = "content1"
    dir_name = "sub_directory"
    inner_file = "b"
    inner_content = "content2"
    local_archive = tempfile.mktemp()

    with tempfile.TemporaryDirectory() as temp_dir:
      with gfile.GFile(os.path.join(temp_dir, outer_file), "w") as f:
        f.write(outer_content)
      directory_path = os.path.join(temp_dir, dir_name)
      os.mkdir(directory_path)
      with gfile.GFile(os.path.join(directory_path, inner_file), "w") as f:
        f.write(inner_content)
      with gfile.Open(local_archive, "wb") as f:
        tfjs_tarfile = tarfile.open(mode="w:gz", fileobj=f)
        tfjs_tarfile.add(temp_dir, arcname="/")
        tfjs_tarfile.close()

    extraction_dir = tempfile.mkdtemp()
    with gfile.GFile(local_archive, "rb") as fileobj:
      file_utils.extract_tarfile_to_destination(fileobj, extraction_dir)

    self.assertCountEqual(
        gfile.ListDirectory(extraction_dir), [outer_file, dir_name])
    self.assertEqual(
        gfile.Open(os.path.join(extraction_dir, outer_file)).read(),
        outer_content)
    self.assertCountEqual(
        gfile.ListDirectory(os.path.join(extraction_dir, dir_name)),
        [inner_file])
    self.assertEqual(
        gfile.Open(os.path.join(extraction_dir, dir_name, inner_file)).read(),
        inner_content)


if __name__ == "__main__":
  tf.test.main()
