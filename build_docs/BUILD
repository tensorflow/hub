load("@rules_python//python:defs.bzl", "py_binary")

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
licenses(["notice"])

package(
    default_applicable_licenses = ["//tensorflow_hub:license"],
    default_visibility = ["//visibility:private"],
)

py_binary(
    name = "build_docs",
    srcs = ["build_docs.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_hub:expect_tensorflow_installed",
        "//third_party/py/tensorflow_docs/api_generator",
        "//tensorflow_hub",
    ],
)
