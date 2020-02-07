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
"""Tests for tensorflow_hub.tfhub_dev.tools.validator."""

import os
import shutil
import tempfile
import tensorflow as tf

from tfhub_dev.tools import validator


class MockFilesystem(validator.Filesystem):
  """Returns a Mock Filesystem storing files in a dictionary."""

  def __init__(self):
    self._files = dict()

  def get_contents(self, filename):
    return self._files[filename]

  def set_contents(self, filename, contents):
    self._files[filename] = contents

  def recursive_list_dir(self, root_dir):
    return [
        f for f in self._files.keys() if f.startswith(root_dir + os.path.sep)
    ]


MINIMAL_MARKDOWN_TEMPLATE = """# Module google/text-embedding-model/1
Simple description spanning
multiple lines.

<!-- asset-path: %s -->
<!-- module-type:   text-embedding   -->
<!-- fine-tunable:true -->
<!-- format: saved_model_2 -->

## Overview
"""

MINIMAL_MARKDOWN_WITH_ALLOWED_LICENSE = """# Module google/model/1
Simple description.

<!-- asset-path: %s -->
<!-- module-type: text-embedding -->
<!-- fine-tunable: true -->
<!-- format: saved_model_2 -->
<!-- license: BSD-3-Clause -->

## Overview
"""

MINIMAL_MARKDOWN_WITH_UNKNOWN_LICENSE = """# Module google/model/1
Simple description.

<!-- asset-path: %s -->
<!-- module-type: text-embedding -->
<!-- fine-tunable: true -->
<!-- format: saved_model_2 -->
<!-- license: my_license -->

## Overview
"""


MARKDOWN_WITHOUT_DESCRIPTION = """# Module google/text-embedding-model/1

<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- format: saved_model_2 -->

## Overview
"""

MARKDOWN_WITH_MISSING_METADATA = """# Module google/text-embedding-model/1
One line description.
<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- format: saved_model_2 -->

## Overview
"""

MARKDOWN_WITH_DUPLICATE_METADATA = """# Module google/text-embedding-model/1
One line description.
<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- asset-path: https://path/to/text-embedding-model/model2.tar.gz -->
<!-- module-type: text-embedding -->
<!-- fine-tunable: true -->
<!-- format: saved_model_2 -->

## Overview
"""

MARKDOWN_WITH_UNEXPECTED_LINES = """# Module google/text-embedding-model/1
One line description.
<!-- module-type: text-embedding -->

This should not be here.
<!-- format: saved_model_2 -->

## Overview
"""

MINIMAL_COLLECTION_MARKDOWN = """# Collection google/text-embedding-collection/1
Simple description spanning
multiple lines.

<!-- module-type: text-embedding -->

## Overview
"""

MINIMAL_PUBLISHER_MARKDOWN = """# Publisher some-publisher
Simple description spanning one line.

[![Icon URL]](https://path/to/icon.png)

## Overview
"""


class ValidatorTest(tf.test.TestCase):

  def setUp(self):
    super(tf.test.TestCase, self).setUp()
    self.tmp_dir = tempfile.mkdtemp()
    self.model_path = os.path.join(self.tmp_dir, "model_1")
    self.not_a_model_path = os.path.join(self.tmp_dir, "not_a_model")
    self.save_dummy_model(self.model_path)
    self.minimal_markdown = MINIMAL_MARKDOWN_TEMPLATE % self.model_path
    self.minimal_markdown_with_bad_model = (
        MINIMAL_MARKDOWN_TEMPLATE % self.not_a_model_path)

  def tearDown(self):
    super(tf.test.TestCase, self).tearDown()
    shutil.rmtree(self.tmp_dir)

  def save_dummy_model(self, path):

    class MultiplyTimesTwoModel(tf.train.Checkpoint):
      """Callable model that multiplies by two."""

      @tf.function(
          input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
      def __call__(self, x):
        return x * 2

    model = MultiplyTimesTwoModel()
    tf.saved_model.save(model, path)

  def test_filesystem(self):
    tmp_dir = self.get_temp_dir()
    tmp_file_path = tempfile.mktemp(dir=tmp_dir)
    file_contents = "CONTENTS"
    with tf.io.gfile.GFile(tmp_file_path, "w") as output_file:
      output_file.write(file_contents)
    filesystem = validator.Filesystem()
    self.assertEqual(file_contents, filesystem.get_contents(tmp_file_path))
    self.assertAllEqual([tmp_file_path],
                        list(filesystem.recursive_list_dir(tmp_dir)))

  def test_minimal_markdown_parsed(self):
    filesystem = MockFilesystem()
    filesystem.set_contents("root/google/models/text-embedding-model/1.md",
                            self.minimal_markdown)
    validator.validate_documentation_files(
        documentation_dir="root", filesystem=filesystem)

  def test_minimal_markdown_parsed_with_selected_files(self):
    filesystem = MockFilesystem()
    filesystem.set_contents("root/google/models/text-embedding-model/1.md",
                            self.minimal_markdown)
    num_validated = validator.validate_documentation_files(
        documentation_dir="root",
        files_to_validate=["google/models/text-embedding-model/1.md"],
        filesystem=filesystem)
    self.assertEqual(1, num_validated)

  def test_minimal_collection_markdown_parsed(self):
    filesystem = MockFilesystem()
    filesystem.set_contents(
        "root/google/collections/text-embedding-collection/1.md",
        MINIMAL_COLLECTION_MARKDOWN)
    validator.validate_documentation_files(
        documentation_dir="root", filesystem=filesystem)

  def test_minimal_publisher_markdown_parsed(self):
    filesystem = MockFilesystem()
    filesystem.set_contents("root/some-publisher/some-publisher.md",
                            MINIMAL_PUBLISHER_MARKDOWN)
    validator.validate_documentation_files(
        documentation_dir="root", filesystem=filesystem)

  def test_invalid_markdown_fails(self):
    filesystem = MockFilesystem()
    filesystem.set_contents("root/publisher/model/1.md", "INVALID MARKDOWN")
    with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                 ".*First line.*"):
      validator.validate_documentation_files(
          documentation_dir="root", filesystem=filesystem)

  def test_minimal_markdown_not_in_publisher_dir(self):
    filesystem = MockFilesystem()
    filesystem.set_contents("root/gooogle/models/wrong-location/1.md",
                            self.minimal_markdown)
    with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                 ".*placed in the publisher directory.*"):
      validator.validate_documentation_files(
          documentation_dir="root", filesystem=filesystem)

  def test_minimal_markdown_does_not_end_with_md_fails(self):
    filesystem = MockFilesystem()
    filesystem.set_contents("root/google/models/wrong-extension/1.mdz",
                            self.minimal_markdown)
    with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                 r".*end with \"\.md.\"*"):
      validator.validate_documentation_files(
          documentation_dir="root", filesystem=filesystem)

  def test_publisher_markdown_at_incorrect_location_fails(self):
    filesystem = MockFilesystem()
    filesystem.set_contents("root/google/publisher.md",
                            MINIMAL_PUBLISHER_MARKDOWN)
    with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                 r".*some-publisher\.md.*"):
      validator.validate_documentation_files(
          documentation_dir="root", filesystem=filesystem)

  def test_publisher_markdown_at_correct_location(self):
    filesystem = MockFilesystem()
    filesystem.set_contents("root/some-publisher/some-publisher.md",
                            MINIMAL_PUBLISHER_MARKDOWN)
    validator.validate_documentation_files(
        documentation_dir="root", filesystem=filesystem)

  def test_markdown_without_description(self):
    filesystem = MockFilesystem()
    filesystem.set_contents("root/google/models/text-embedding-model/1.md",
                            MARKDOWN_WITHOUT_DESCRIPTION)
    with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                 ".*has to contain a short description.*"):
      validator.validate_documentation_files(
          documentation_dir="root", filesystem=filesystem)

  def test_markdown_with_missing_metadata(self):
    filesystem = MockFilesystem()
    filesystem.set_contents("root/google/models/text-embedding-model/1.md",
                            MARKDOWN_WITH_MISSING_METADATA)
    with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                 ".*missing.*fine-tunable.*module-type.*"):
      validator.validate_documentation_files(
          documentation_dir="root", filesystem=filesystem)

  def test_markdown_with_duplicate_metadata(self):
    filesystem = MockFilesystem()
    filesystem.set_contents("root/google/models/text-embedding-model/1.md",
                            MARKDOWN_WITH_DUPLICATE_METADATA)
    with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                 ".*duplicate.*asset-path.*"):
      validator.validate_documentation_files(
          documentation_dir="root", filesystem=filesystem)

  def test_markdown_with_unexpected_lines(self):
    filesystem = MockFilesystem()
    filesystem.set_contents("root/google/models/text-embedding-model/1.md",
                            MARKDOWN_WITH_UNEXPECTED_LINES)
    with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                 ".*Unexpected line.*"):
      validator.validate_documentation_files(
          documentation_dir="root", filesystem=filesystem)

  def test_minimal_markdown_parsed_full(self):
    documentation_parser = validator.DocumentationParser("root")
    documentation_parser.validate(
        file_path="root/google/models/text-embedding-model/1.md",
        documentation_content=self.minimal_markdown,
        do_smoke_test=True)
    self.assertEqual("Simple description spanning multiple lines.",
                     documentation_parser.parsed_description)
    expected_metadata = {
        "asset-path": {self.model_path},
        "module-type": {"text-embedding"},
        "fine-tunable": {"true"},
        "format": {"saved_model_2"},
    }
    self.assertAllEqual(expected_metadata, documentation_parser.parsed_metadata)

  def test_bad_model_does_not_pass_smoke_test(self):
    filesystem = MockFilesystem()
    filesystem.set_contents("root/google/models/text-embedding-model/1.md",
                            self.minimal_markdown_with_bad_model)
    with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                 ".*failed to parse.*"):
      validator.validate_documentation_files(
          documentation_dir="root",
          files_to_validate=["google/models/text-embedding-model/1.md"],
          filesystem=filesystem)

  def test_markdown_with_allowed_license(self):
    filesystem = MockFilesystem()
    filesystem.set_contents("root/google/models/model/1.md",
                            MINIMAL_MARKDOWN_WITH_ALLOWED_LICENSE)
    validator.validate_documentation_files(
        documentation_dir="root", filesystem=filesystem)

  def test_markdown_with_unknown_license(self):
    filesystem = MockFilesystem()
    filesystem.set_contents("root/google/models/model/1.md",
                            MINIMAL_MARKDOWN_WITH_UNKNOWN_LICENSE)
    with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                 ".*specify a license id from list.*"):
      validator.validate_documentation_files(
          documentation_dir="root", filesystem=filesystem)

if __name__ == "__main__":
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
