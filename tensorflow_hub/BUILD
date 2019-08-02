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
licenses(["notice"])  # Apache 2.0 License

package(default_visibility = ["//:__subpackages__"])

exports_files(["LICENSE"])

load("//tensorflow_hub:protos.bzl", "tf_hub_proto_library")

# This is the public import users should use.
py_library(
    name = "tensorflow_hub",
    srcs = [
        "__init__.py",
        "version.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":estimator",
        ":feature_column",
        ":image_util",
        ":keras_layer",
        ":module",
        ":module_v2",
        ":saved_model_module",
        "//tensorflow_hub:expect_tensorflow_installed",
        ":config",
    ],
)

py_library(
    name = "tf_v1",
    srcs = [
        "tf_v1.py",
    ],
    deps = [
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_library(
    name = "config",
    srcs = ["config.py"],
    deps = [
        ":compressed_module_resolver",
        ":native_module",
        ":registry",
        ":resolver",
    ],
)

py_library(
    name = "estimator",
    srcs = ["estimator.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tf_utils",
        ":tf_v1",
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_test(
    name = "estimator_test",
    srcs = ["estimator_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    tags = ["notsan"],  # b/111624588
    deps = [
        "//tensorflow_hub:expect_tensorflow_installed",
        "//tensorflow_hub",
    ],
)

py_library(
    name = "feature_column",
    srcs = ["feature_column.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":module",
        ":tf_v1",
        "//tensorflow_hub:expect_tensorflow_installed",
        #  "tensorflow/python/feature_column",
    ],
)

py_test(
    name = "feature_column_test",
    srcs = ["feature_column_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":tensorflow_hub",
        ":test_utils",
        "//tensorflow_hub:expect_numpy_installed",
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_library(
    name = "compressed_module_resolver",
    srcs = ["compressed_module_resolver.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":resolver",
        ":tf_v1",
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_test(
    name = "compressed_module_resolver_test",
    size = "medium",
    srcs = ["compressed_module_resolver_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    tags = ["nofixdeps"],
    deps = [
        ":compressed_module_resolver",
        ":tensorflow_hub",
        ":test_utils",
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_library(
    name = "image_util",
    srcs = ["image_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":all_protos_py_pb2",
        ":native_module",
    ],
)

py_test(
    name = "image_util_test",
    srcs = ["image_util_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":image_util",
        ":module",
        ":native_module",
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_library(
    name = "module",
    srcs = [
        "module.py",
        "module_impl.py",
        "module_spec.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":registry",
        ":tensor_info",
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_test(
    name = "module_test",
    srcs = ["module_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":module",
        ":tensor_info",
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_library(
    name = "saved_model_module",
    srcs = ["saved_model_module.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":module",
        ":native_module",
        ":saved_model_lib",
        ":tf_utils",
        ":tf_v1",
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_test(
    name = "saved_model_module_test",
    srcs = ["saved_model_module_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":saved_model_lib",
        ":saved_model_module",
        ":tensorflow_hub",
        ":tf_utils",
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_library(
    name = "native_module",
    srcs = ["native_module.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":all_protos_py_pb2",
        ":meta_graph_lib",
        ":module",
        ":saved_model_lib",
        ":tf_utils",
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_test(
    name = "native_module_test",
    srcs = ["native_module_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":all_protos_py_pb2",
        ":native_module",
        ":tensorflow_hub",
        "//tensorflow_hub:expect_numpy_installed",
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_library(
    name = "resolver",
    srcs = ["resolver.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tf_utils",
        ":tf_v1",
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_test(
    name = "resolver_test",
    size = "medium",
    srcs = ["resolver_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":resolver",
        ":tf_utils",
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    srcs_version = "PY2AND3",
    tags = ["nofixdeps"],
    deps = [
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_test(
    name = "e2e_test",
    srcs = ["e2e_test.py"],
    data = [
        "testdata/half_plus_two_v1.tar.gz",
    ],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":tensorflow_hub",
        ":test_utils",
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_library(
    name = "saved_model_lib",
    srcs = ["saved_model_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":all_protos_py_pb2",
        ":tf_utils",
        ":tf_v1",
        "//tensorflow_hub:expect_tensorflow_installed",
        #  "tensorflow/core:protos_all_py_pb2",
    ],
)

py_test(
    name = "saved_model_lib_test",
    size = "small",
    srcs = ["saved_model_lib_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":saved_model_lib",
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_library(
    name = "meta_graph_lib",
    srcs = ["meta_graph_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_hub:expect_tensorflow_installed",
        #  "tensorflow/core:protos_all_py_pb2",
    ],
)

py_library(
    name = "tensor_info",
    srcs = ["tensor_info.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tf_v1",
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_test(
    name = "tensor_info_test",
    size = "small",
    srcs = ["tensor_info_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":tensor_info",
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_library(
    name = "tf_utils",
    srcs = ["tf_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_library(
    name = "registry",
    srcs = ["registry.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "registry_test",
    srcs = ["registry_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":registry",
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_library(
    name = "keras_layer",
    srcs = ["keras_layer.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

py_test(
    name = "keras_layer_test",
    srcs = ["keras_layer_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":tensorflow_hub",
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)

tf_hub_proto_library(
    name = "all_protos",
    srcs = [
        "image_module_info.proto",
        "module_attachment.proto",
        "module_def.proto",
    ],
    visibility = ["//:__subpackages__"],
)

# We expect TensorFlow to already be installed on the system, e.g. via
# `pip install tensorflow`
py_library(
    name = "expect_tensorflow_installed",
)

# We expect numpy to already be installed on the system, e.g. via
# `pip install numpy`
py_library(
    name = "expect_numpy_installed",
)

# We expect numpy to already be installed on the system, e.g. via
# `pip install tensorflow_datasets`
py_library(
    name = "expect_tensorflow_datasets_installed",
)

py_library(
    name = "module_v2",
    srcs = ["module_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":module",
        ":native_module",
        ":tf_v1",
        "//tensorflow_hub:expect_tensorflow_installed",
    ],
)
