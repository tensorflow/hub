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
load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

def tf_hub_proto_library(name = None, srcs = [], visibility = []):
    py_proto_library(
        name = name + "_py_pb2",
        srcs = srcs,
        srcs_version = "PY2AND3",
        visibility = visibility,
        # We pull in @com_google_protobuf to get protoc, but we want the
        # generated Python code to pick up its runtime via
        # `pip install protobuf`, same as if using `pip install tensorflow_hub`.
        default_runtime = "//tensorflow_hub:expect_protobuf_installed",
        protoc = "@com_google_protobuf//:protoc",
    )
