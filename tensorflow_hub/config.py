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
"""Configuration to bind implementations on the API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_hub import compressed_module_resolver
from tensorflow_hub import native_module
from tensorflow_hub import registry
from tensorflow_hub import resolver


def _install_default_resolvers():
  for impl in [
      resolver.PathResolver(),
      compressed_module_resolver.GcsCompressedFileResolver(),
      compressed_module_resolver.HttpCompressedFileResolver()
  ]:
    registry.resolver.add_implementation(impl)


def _run():
  _install_default_resolvers()

  registry.loader.add_implementation(native_module.Loader())
