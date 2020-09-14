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
licenses(["notice"])  # Apache 2.0

py_binary(
    name = "validator",
    srcs = ["validator.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":validator_lib"],
)

py_library(
    name = "validator_lib",
    srcs = [
        "validator.py",
    ],
    srcs_version = "PY3",
    visibility = ["//learning/brain/contrib/hub/tools/repository:__pkg__"],
    deps = [
        "//tensorflow_hub:expect_tensorflow_installed",
        "//tensorflow_hub",
    ],
)

py_test(
    name = "validator_test",
    srcs = ["validator_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":validator_lib",
        "//tensorflow_hub:expect_tensorflow_installed",
        "//tensorflow_hub",
    ],
)
