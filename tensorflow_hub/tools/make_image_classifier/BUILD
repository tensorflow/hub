# Copyright 2019 The TensorFlow Hub Authors. All Rights Reserved.
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
licenses(["notice"])  # Apache 2.0 License

package(
    default_visibility = [
        "//:__subpackages__",
        "//tensorflow_hub:__subpackages__",
    ],
)

# A library with reusable pieces of make_image_classifier.
py_library(
    name = "make_image_classifier_lib",
    srcs = ["make_image_classifier_lib.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_hub:expect_tensorflow_installed",
        "//tensorflow_hub",
    ],
)

# The make_image_classifier script as a py_library (private, for testing only).
py_library(
    name = "make_image_classifier_main",
    srcs = ["make_image_classifier.py"],
    srcs_version = "PY3",
    deps = [
        ":make_image_classifier_lib",
        "//tensorflow_hub:expect_tensorflow_installed",
        "//tensorflow_hub",
    ],
)

# The make_image_classifier script as a py_binary.
py_binary(
    name = "make_image_classifier",
    srcs = ["make_image_classifier.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":make_image_classifier_main",
    ],
)

py_test(
    name = "make_image_classifier_test",
    srcs = ["make_image_classifier_test.py"],
    python_version = "PY3",
    deps = [
        ":make_image_classifier_lib",
        ":make_image_classifier_main",
        "//tensorflow_hub:expect_absl_py_installed",  # "/testing:flagsaver"
        "//tensorflow_hub:expect_tensorflow_installed",
        "//tensorflow_hub",
    ],
)
