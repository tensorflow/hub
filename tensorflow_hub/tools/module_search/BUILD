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

py_binary(
    name = "search",
    srcs = ["search.py"],
    python_version = "PY3",
    deps = [
        ":search_lib",
        ":utils",
        "//tensorflow_hub:expect_numpy_installed",
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_library(
    name = "search_lib",
    srcs = ["search.py"],
    deps = [
        ":utils",
        "//tensorflow_hub:expect_numpy_installed",
        "//tensorflow_hub:expect_pandas_installed",
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//tensorflow_hub:expect_numpy_installed",
        "//tensorflow_hub:expect_tensorflow_installed",
        "//tensorflow_hub:expect_tensorflow_datasets_installed",
        "//tensorflow_hub",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    python_version = "PY3",
    tags = [
        "noasan",  # sklearn seem to lead to asan and tsan failures.
        "notsan",
    ],
    deps = [
        ":utils",
        "//tensorflow_hub:expect_sklearn_installed",
    ],
)

py_test(
    name = "search_test",
    srcs = ["search_test.py"],
    python_version = "PY3",
    deps = [
        ":search_lib",
        "//tensorflow_hub:expect_absl_py_installed",  # "/testing:flagsaver"
        "//tensorflow_hub:expect_tensorflow_installed",
        "//tensorflow_hub:expect_tensorflow_datasets_installed",
    ],
)
