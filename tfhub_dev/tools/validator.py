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
"""Markdown documentation validator for published models.

1) To validate selected files, run from the project root path:
$ python tfhub_dev/tools/validator.py vtab/models/wae-ukl/1.md [other_files]

This will download and smoke test the model specified on asset-path metadata.

2) To validate all documentation files, run from the project root path:
$ python tfhub_dev/tools/validator.py

This does not download and smoke test the model.

3) To validate files from outside the project root path, use the --root_dir
flag:
$ python tfhub_dev/tools/validator.py --root_dir=path_to_project_root
"""

import abc
import argparse
import os
import re
import sys
from absl import app
from absl import logging

import tensorflow as tf
import tensorflow_hub as hub

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.saved_model import loader_impl
# pylint: enable=g-direct-tensorflow-import

FLAGS = None

# Regex pattern for the first line of the documentation of models.
# Example: "Module google/universal-sentence-encoder/1"
MODEL_HANDLE_PATTERN = (
    r"# Module (?P<publisher>[\w-]+)/(?P<name>(\w|-|/|&|;|\.)+)/(?P<vers>\d+)")
# Regex pattern for the first line of the documentation of publishers.
# Example: "Publisher google"
PUBLISHER_HANDLE_PATTERN = r"# Publisher (?P<publisher>[\w-]+)"
# Regex pattern for the first line of the documentation of collections.
# Example: "Collection google/universal-sentence-encoders/1"
COLLECTION_HANDLE_PATTERN = (
    r"# Collection (?P<publisher>[\w-]+)/(?P<name>(\w|-|/|&|;|\.)+)/(\d+)")
# Regex pattern for the line of the documentation describing model metadata.
# Example: "<!-- finetunable: true -->"
# Note: Both key and value consumes free space characters, but later on these
# are stripped.
METADATA_LINE_PATTERN = r"^<!--(?P<key>(\w|\s|-)+):(?P<value>.+)-->$"


class Filesystem(object):
  """Convenient (and mockable) file system access."""

  def get_contents(self, filename):
    """Returns file contents as a string."""
    with tf.io.gfile.GFile(filename, "r") as f:
      return f.read()

  def file_exists(self, filename):
    """Returns whether file exists."""
    return tf.io.gfile.exists(filename)

  def recursive_list_dir(self, root_dir):
    """Yields all files of a root directory tree."""
    for dirname, _, filenames in tf.io.gfile.walk(root_dir):
      for filename in filenames:
        yield os.path.join(dirname, filename)


class MarkdownDocumentationError(Exception):
  """Problem with markdown syntax parsing."""


def smoke_test_model(model_path):
  try:
    resolved_model = hub.resolve(model_path)
    loader_impl.parse_saved_model(resolved_model)
  except Exception as e:  # pylint: disable=broad-except
    return False, e
  return True, None


class ParsingPolicy(object):
  """The base class for type specific parsing policy."""
  __metaclass__ = abc.ABCMeta

  def __init__(self, publisher, model_name, model_version):
    self._publisher = publisher
    self._model_name = model_name
    self._model_version = model_version

  @property
  @abc.abstractmethod
  def type_name(self):
    """Return readable name of the parsed type."""

  @property
  def handle(self):
    return "%s/%s/%s" % (self._publisher, self._model_name, self._model_version)

  @property
  def publisher(self):
    return self._publisher

  def get_expected_location(self, root_dir):
    """Returns the expected path of a documentation file."""
    del root_dir
    return

  def get_top_level_dir(self, root_dir):
    """Returns the top level publisher directory."""
    return os.path.join(root_dir, self._publisher)

  def get_required_metadata(self):
    """Return a list of required metadata for this type."""
    return list()

  def get_single_valued_metadata(self):
    """Return a list of metadata attaining only one value for this type."""
    return list()

  def asset_tester(self):
    """Return a function that smoke tests an asset.

    This function takes asset path on input and returns a tuple
    (passed, reason), where passed==True iff the asset passes a smoke test and
    reason is None for passed==True, or reason for failing if passed==False.

    Returns:
      A function that smoke tests an asset.
    """
    return lambda _: True, None


class ModelParsingPolicy(ParsingPolicy):
  """ParsingPolicy for model documentation."""

  def type_name(self):
    return "Module"

  def get_required_metadata(self):
    return ["asset-path", "format", "module-type", "fine-tunable"]

  def get_single_valued_metadata(self):
    return ["asset-path", "format", "module-type", "fine-tunable"]

  def asset_tester(self):
    return smoke_test_model


class PublisherParsingPolicy(ParsingPolicy):
  """ParsingPolicy for publisher documentation."""

  def type_name(self):
    return "Publisher"

  @property
  def handle(self):
    return self._publisher

  def get_expected_location(self, root_dir):
    return os.path.join(root_dir, self._publisher, self._publisher + ".md")


class CollectionParsingPolicy(ParsingPolicy):
  """ParsingPolicy for collection documentation."""

  def type_name(self):
    return "Collection"

  def get_required_metadata(self):
    return ["module-type"]


class DocumentationParser(object):
  """Class used for parsing model documentation strings."""

  def __init__(self, documentation_dir, filesystem):
    self._documentation_dir = documentation_dir
    self._filesystem = filesystem
    self._parsed_metadata = dict()
    self._parsed_description = ""

  @property
  def parsed_description(self):
    return self._parsed_description

  @property
  def parsed_metadata(self):
    return self._parsed_metadata

  def raise_error(self, message):
    message_with_file = "Error at file %s: %s" % (self._file_path, message)
    raise MarkdownDocumentationError(message_with_file)

  def consume_first_line(self):
    """Consume first line describing the model type and handle."""
    first_line = self._lines[0].replace("&zwnj;", "")
    patterns_and_policies = [
        (MODEL_HANDLE_PATTERN, ModelParsingPolicy),
        (PUBLISHER_HANDLE_PATTERN, PublisherParsingPolicy),
        (COLLECTION_HANDLE_PATTERN, CollectionParsingPolicy),
    ]
    for pattern, policy in patterns_and_policies:
      match = re.match(pattern, first_line)
      if not match:
        continue
      groups = match.groupdict()
      self._parsing_policy = policy(
          groups.get("publisher"), groups.get("name"), groups.get("vers"))
      return
    self.raise_error(
        "First line of the documentation file must describe either the model "
        "handle in format \"%s\", or a publisher handle in format \"%s\", or "
        "a collection handle in format \"%s\". For example "
        "\"# Module google/text-embedding-model/1\". Instead the first line "
        "is \"%s\"." % (MODEL_HANDLE_PATTERN, PUBLISHER_HANDLE_PATTERN,
                        COLLECTION_HANDLE_PATTERN, first_line))

  def assert_publisher_page_exists(self):
    """Assert that publisher page exists for the publisher of this model."""
    # Use a publisher policy to get the expected documentation page path.
    publisher_policy = PublisherParsingPolicy(self._parsing_policy.publisher,
                                              self._parsing_policy.publisher,
                                              None)
    expected_publisher_doc_location = publisher_policy.get_expected_location(
        self._documentation_dir)
    if not self._filesystem.file_exists(expected_publisher_doc_location):
      self.raise_error(
          "Publisher documentation does not exist. It should be added to %s." %
          expected_publisher_doc_location)

  def assert_correct_location(self):
    """Assert that documentation file is submitted to a correct location."""
    expected_file_path = self._parsing_policy.get_expected_location(
        self._documentation_dir)
    # Exact location must be enforced for some types (publishers).
    if expected_file_path and self._file_path != expected_file_path:
      self.raise_error(
          "Documentation file is not on a correct path. Documentation for a "
          "%s with handle \"%s\" should be submitted to \"%s\" " %
          (self._parsing_policy.type_name, self._parsing_policy.handle,
           expected_file_path))

    publisher_dir = self._parsing_policy.get_top_level_dir(
        self._documentation_dir)
    if not self._file_path.startswith(publisher_dir + "/"):
      self.raise_error(
          "Documentation file is not on a correct path. Documentation for a "
          "%s with handle \"%s\" should be placed in the publisher "
          "directory: \"%s\"" % (self._parsing_policy.type_name,
                                 self._parsing_policy.handle, publisher_dir))

    if not self._file_path.endswith(".md"):
      self.raise_error("Documentation file does not end with \".md\": %s" %
                       self._file_path)

  def consume_description(self):
    """Consume second line with a short model description."""
    first_description_line = self._lines[1]
    if not first_description_line:
      self.raise_error(
          "Second line of the documentation file has to contain a short "
          "description. For example \"Word2vec text embedding model.\".")
    self._parsed_description = self._lines[1]
    self._current_index = 2
    while self._lines[self._current_index] and not self._lines[
        self._current_index].startswith("<!--"):
      self._parsed_description += " " + self._lines[self._current_index]
      self._current_index += 1

  def consume_metadata(self):
    """Consume all metadata."""
    while not self._lines[self._current_index].startswith("#"):
      if not self._lines[self._current_index]:
        # Empty line is ok.
        self._current_index += 1
        continue
      match = re.match(METADATA_LINE_PATTERN, self._lines[self._current_index])
      if match:
        # Add found metadata.
        groups = match.groupdict()
        key = groups.get("key").strip()
        value = groups.get("value").strip()
        if key not in self._parsed_metadata:
          self._parsed_metadata[key] = set()
        self._parsed_metadata[key].add(value)
        self._current_index += 1
        continue
      if self._lines[self._current_index].startswith("[![Icon URL]]"):
        # Icon for publishers.
        self._current_index += 1
        continue
      if self._lines[self._current_index].startswith(
          "[![Open Colab notebook]]"):
        # Colab.
        self._current_index += 1
        continue
      # Not an empty line and not expected metadata.
      self.raise_error(
          "Unexpected line found: \"%s\". Please refer to [README.md]"
          "(https://github.com/tensorflow/hub/blob/master/tensorflow_hub"
          "/tfhub_dev/README.md) for information about markdown format." %
          self._lines[self._current_index])

  def assert_correct_metadata(self):
    """Assert that all required metadata is present."""
    required_metadata = set(self._parsing_policy.get_required_metadata())
    provided_metadata = set(self._parsed_metadata.keys())
    if not provided_metadata.issuperset(required_metadata):
      self.raise_error(
          "There are missing required metadata lines. Please refer to "
          "README.md for information about markdown format. In particular the "
          "missing metadata are: %s" %
          sorted(required_metadata.difference(provided_metadata)))

    duplicate_metadata = list()
    for key, values in self._parsed_metadata.items():
      if key in self._parsing_policy.get_single_valued_metadata(
      ) and len(values) > 1:
        duplicate_metadata.append(key)
    if duplicate_metadata:
      self.raise_error(
          "There are duplicate metadata values. Please refer to "
          "README.md for information about markdown format. In particular the "
          "duplicated metadata are: %s" % sorted(duplicate_metadata))

    if "module-type" in self._parsed_metadata:
      allowed_prefixes = ["image-", "text-", "audio-", "video-"]
      for value in self._parsed_metadata["module-type"]:
        if all([not value.startswith(prefix) for prefix in allowed_prefixes]):
          self.raise_error(
              "The \"module-type\" metadata has to start with any of \"image-\""
              ", \"text\", \"audio-\", \"video-\", but is: \"%s\"" % value)

  def assert_allowed_license(self):
    """Validate provided license."""
    if "license" in self._parsed_metadata:
      license_id = list(self._parsed_metadata["license"])[0]
      allowed_license_ids = [
          "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "GPL-2.0", "GPL-3.0",
          "LGPL-2.0", "LGPL-2.1", "LGPL-3.0", "MIT", "MPL-2.0", "CDDL-1.0",
          "EPL-2.0", "custom"
      ]
      if license_id not in allowed_license_ids:
        self.raise_error(
            "The license %s provided in metadata is not allowed. Please "
            "specify a license id from list of allowed ids: [%s]. Example: "
            "<!-- license: Apache-2.0 -->" % (license_id, allowed_license_ids))

  def smoke_test_asset(self):
    """Smoke test asset provided on asset-path metadata."""
    if "asset-path" in self._parsed_metadata:
      asset_path = list(self._parsed_metadata["asset-path"])[0]
      asset_tester = self._parsing_policy.asset_tester()
      passed, reason = asset_tester(asset_path)
      if not passed:
        self.raise_error(
            "The model on path %s failed to parse. Please make sure that the "
            "asset-path metadata points to a valid TF2 SavedModel or a TF1 Hub "
            "module, compressed as described in section \"Model\" of "
            "README.md. Underlying reason for failure: %s." %
            (asset_path, reason))

  def validate(self, file_path, do_smoke_test):
    """Validate one documentation markdown file."""
    self._raw_content = self._filesystem.get_contents(file_path)
    self._lines = self._raw_content.split("\n")
    self._file_path = file_path
    self.consume_first_line()
    self.assert_correct_location()
    self.consume_description()
    self.consume_metadata()
    self.assert_correct_metadata()
    self.assert_allowed_license()
    self.assert_publisher_page_exists()
    if do_smoke_test:
      self.smoke_test_asset()


def validate_documentation_files(documentation_dir,
                                 files_to_validate=None,
                                 filesystem=Filesystem()):
  """Validate documentation files in a directory."""
  file_paths = list(filesystem.recursive_list_dir(documentation_dir))
  do_smoke_test = bool(files_to_validate)
  validated = 0
  for file_path in file_paths:
    if files_to_validate and file_path[len(documentation_dir) +
                                       1:] not in files_to_validate:
      continue
    logging.info("Validating %s.", file_path)
    documentation_parser = DocumentationParser(documentation_dir, filesystem)
    documentation_parser.validate(file_path, do_smoke_test)
    validated += 1
  logging.info("Found %d matching files - all validated successfully.",
               validated)
  if not do_smoke_test:
    logging.info(
        "No models were smoke tested. To download and smoke test a specific "
        "model, specify files directly in the command line, for example: "
        "\"python tfhub_dev/tools/validator.py vtab/models/wae-ukl/1.md\"")
  return validated


def main(_):
  root_dir = FLAGS.root_dir or os.getcwd()
  documentation_dir = os.path.join(root_dir, "tfhub_dev", "assets")
  logging.info("Using %s for documentation directory.", documentation_dir)

  files_to_validate = None
  if FLAGS.file:
    files_to_validate = FLAGS.file
    logging.info("Going to validate files %s in documentation directory %s.",
                 files_to_validate, documentation_dir)
  else:
    logging.info("Going to validate all files in documentation directory %s.",
                 documentation_dir)

  validate_documentation_files(
      documentation_dir=documentation_dir, files_to_validate=files_to_validate)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "file",
      type=str,
      default=None,
      help=("Path to files to validate. Path is relative to `--root_dir`. "
            "The model will be smoke tested only for files specified by this "
            "flag."),
      nargs="*")
  parser.add_argument(
      "--root_dir",
      type=str,
      default=None,
      help=("Root directory that contains documentation files under "
            "./tfhub_dev/assets. Defaults to current directory."))
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
