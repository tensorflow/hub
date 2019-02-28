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
"""TensorFlow Hub Module API for Tensorflow 2.0"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_hub import native_module
from tensorflow_hub import registry
from tensorflow_hub import tf_v1


def resolve(handle):
  """Resolves a module handle into a path.

   Resolves a module handle into a path by downloading and caching in
   location specified by TF_HUB_CACHE_DIR if needed.

  Args:
    handle: (string) the Module handle to resolve.

  Returns:
    A string representing the Module path.
  """
  return registry.resolver(handle)


def load(handle):
  """Loads a module from a handle.

  Currently this method only works with Tensorflow 2.x and can only load modules
  created by calling tensorflow.saved_model.save(). The method works in both
  eager and graph modes.

  Depending on the type of handle used, the call may involve downloading a
  Tensorflow Hub module to a local cache location specified by the
  TFHUB_CACHE_DIR environment variable. If a copy of the module is already
  present in the TFHUB_CACHE_DIR, the download step is skipped.

  Currently, three types of module handles are supported:
    1) Smart URL resolvers such as tfhub.dev, e.g.:
       https://tfhub.dev/google/nnlm-en-dim128/1.
    2) A directory on a file system supported by Tensorflow containing module
       files. This may include a local directory (e.g. /usr/local/mymodule) or a
       Google Cloud Storage bucket (gs://mymodule).
    3) A URL pointing to a TGZ archive of a module, e.g.
       https://example.com/mymodule.tar.gz.

  Args:
    handle: (string) the Module handle to resolve.

  Returns:
    A trackable object (see tf.saved_model.load() documentation for details).

  Raises:
    NotImplementedError: If the code is running against incompatible (1.x)
                         version of TF.
  """
  if hasattr(tf_v1.saved_model, "load_v2"):
    module_handle = resolve(handle)
    if tf_v1.gfile.Exists(native_module.get_module_proto_path(module_handle)):
      raise NotImplementedError("TF Hub module '%s' is stored using TF 1.x "
                                "format. Loading of the module using "
                                "hub.load() is not supported." % handle)
    return tf_v1.saved_model.load_v2(module_handle)
  else:
    raise NotImplementedError("hub.load() is not implemented for TF < 1.14.x, "
                              "Current version: %s", tf.__version__)
