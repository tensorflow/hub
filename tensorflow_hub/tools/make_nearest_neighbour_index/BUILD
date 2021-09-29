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

# A library for embedding_generator.
py_library(
    name = "embedding_generator",
    srcs = ["embedding_generator.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_hub:expect_apache_beam_installed",
        "//tensorflow_hub:expect_sklearn_installed",
        "//tensorflow_hub:expect_tensorflow_installed",
        "//tensorflow_hub",
    ],
)

# A library for index_builder.
py_library(
    name = "index_builder",
    srcs = ["index_builder.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_hub:expect_annoy_installed",
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

# A library for similarity_finder.
py_library(
    name = "similarity_finder",
    srcs = ["similarity_finder.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_hub:expect_annoy_installed",
        "//tensorflow_hub:expect_tensorflow_installed",
        "//tensorflow_hub",
    ],
)

# The make_nearest_neighbour_index script as a py_binary.
py_binary(
    name = "make_nearest_neighbour_index",
    srcs = ["make_nearest_neighbour_index.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":embedding_generator",
        ":index_builder",
        ":similarity_finder",
        "//tensorflow_hub:expect_absl_py_installed",  # ":app"
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_test(
    name = "embedding_generator_test",
    srcs = ["embedding_generator_test.py"],
    data = [
        "test_data/data/titles.txt",
    ],
    python_version = "PY3",
    tags = ["requires-net:external"],
    deps = [
        ":embedding_generator",
        "//tensorflow_hub:expect_resources_installed",
        "//tensorflow_hub:expect_absl_py_installed",  # "/flags"
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_test(
    name = "index_builder_test",
    srcs = ["index_builder_test.py"],
    data = [
        "test_data/embeds/emb-00000-of-00001.tfrecords",
        "test_data/embeds/random_projection.matrix",
    ],
    python_version = "PY3",
    deps = [
        ":index_builder",
        "//tensorflow_hub:expect_resources_installed",
        "//tensorflow_hub:expect_absl_py_installed",  # "/flags"
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)
