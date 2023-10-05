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
workspace(name = "org_tensorflow_hub")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Needed by com_google_protobuf.
http_archive(
    name = "rules_python",  # 2023-01-10T22:00:51Z
    sha256 = "5de54486a60ad8948dabe49605bb1c08053e04001a431ab3e96745b4d97a4419",
    strip_prefix = "rules_python-70cce26432187a60b4e950118791385e6fb3c26f",
    urls = ["https://github.com/bazelbuild/rules_python/archive/70cce26432187a60b4e950118791385e6fb3c26f.zip"],
)

# For use by //tensorflow_hub:protos.bzl.
# 3.19.x is the minimum possible protoc that will generate compiled proto code that is
# compatible with protobuf runtimes >= 4, as discussed here:
# https://developers.google.com/protocol-buffers/docs/news/2022-05-06
http_archive(
    name = "com_google_protobuf",
    sha256 = "9a301cf94a8ddcb380b901e7aac852780b826595075577bb967004050c835056",
    strip_prefix = "protobuf-3.19.6",
    urls = [
        "http://mirror.tensorflow.org/github.com/protocolbuffers/protobuf/archive/v3.19.6.tar.gz",
        "https://github.com/protocolbuffers/protobuf/archive/v3.19.6.tar.gz",  # 2022-09-29
    ],
)

# Required by protobuf 3.19.6.
http_archive(
    name = "zlib",
    build_file = "@com_google_protobuf//:third_party/zlib.BUILD",
    sha256 = "b3a24de97a8fdbc835b9833169501030b8977031bcb54b3b3ac13740f846ab30",
    strip_prefix = "zlib-1.2.13",
    urls = [
      "https://storage.googleapis.com/mirror.tensorflow.org/zlib.net/zlib-1.2.13.tar.gz",
      "https://zlib.net/zlib-1.2.13.tar.gz",
      ],
)

http_archive(
  name = "rules_license",
  sha256 = "6157e1e68378532d0241ecd15d3c45f6e5cfd98fc10846045509fb2a7cc9e381",
  urls = [
    "https://mirror.bazel.build/github.com/bazelbuild/rules_license/releases/download/0.0.4/rules_license-0.0.4.tar.gz",
    "https://github.com/bazelbuild/rules_license/releases/download/0.0.4/rules_license-0.0.4.tar.gz",
  ],
)

# Required by protobuf 3.19.6.
http_archive(
    name = "bazel_skylib",
    sha256 = "f7be3474d42aae265405a592bb7da8e171919d74c16f082a5457840f06054728",  # Last updated 2022-05-18
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/1.2.1/bazel-skylib-1.2.1.tar.gz"],
)
