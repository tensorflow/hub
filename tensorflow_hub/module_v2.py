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
"""TensorFlow Hub Module API for Tensorflow 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import six

from tensorflow_hub import native_module
from tensorflow_hub import registry
from tensorflow_hub import tf_v1


def resolve(handle):
  """Resolves a module handle into a path.

  This function works both for plain TF2 SavedModels and the legacy TF1 Hub
  format.

  Resolves a module handle into a path by downloading and caching in
  location specified by TF_HUB_CACHE_DIR if needed.

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
    A string representing the Module path.
  """
  return registry.resolver(handle)


def load(handle, tags=None, options=None):
  """Resolves a handle and loads the resulting module.

  This is the preferred API to load a Hub module in low-level TensorFlow 2.
  Users of higher-level frameworks like Keras should use the framework's
  corresponding wrapper, like hub.KerasLayer.

  This function is roughly equivalent to the TF2 function
  `tf.saved_model.load()` on the result of `hub.resolve(handle)`. Calling this
  function requires TF 1.14 or newer. It can be called both in eager and graph
  mode.

  Note: Using in a tf.compat.v1.Session with variables placed on parameter
  servers requires setting `experimental.share_cluster_devices_in_session`
  within the `tf.compat.v1.ConfigProto`. (It becomes non-experimental in TF2.2.)

  This function can handle the deprecated TF1 Hub format to the extent
  that `tf.saved_model.load()` in TF2 does. In particular, the returned object
  has attributes
    * `.variables`: a list of variables from the loaded object;
    * `.signatures`: a dict of TF2 ConcreteFunctions, keyed by signature names,
      that take tensor kwargs and return a tensor dict.
  However, the information imported by hub.Module into the collections of a
  tf.Graph is lost (e.g., regularization losses and update ops).

  Args:
    handle: (string) the Module handle to resolve; see hub.resolve().
    tags: A set of strings specifying the graph variant to use, if loading from
      a v1 module.
    options: Optional, `tf.saved_model.LoadOptions` object that specifies
      options for loading. This argument can only be used from TensorFlow 2.3
      onwards.

  Returns:
    A trackable object (see tf.saved_model.load() documentation for details).

  Raises:
    NotImplementedError: If the code is running against incompatible (1.x)
                         version of TF.
  """
  if not hasattr(tf_v1.saved_model, "load_v2"):
    raise NotImplementedError("hub.load() is not implemented for TF < 1.14.x, "
                              "Current version: %s" % tf.__version__)
  if not isinstance(handle, six.string_types):
    raise ValueError("Expected a string, got %s" % handle)
  module_path = resolve(handle)
  is_hub_module_v1 = tf.io.gfile.exists(
      native_module.get_module_proto_path(module_path))
  if tags is None and is_hub_module_v1:
    tags = []

  if options:
    if not hasattr(getattr(tf, "saved_model", None), "LoadOptions"):
      raise NotImplementedError("options are not supported for TF < 2.3.x,"
                                " Current version: %s" % tf.__version__)
    obj = tf_v1.saved_model.load_v2(
        module_path, tags=tags, options=options)
  else:
    obj = tf_v1.saved_model.load_v2(module_path, tags=tags)
  obj._is_hub_module_v1 = is_hub_module_v1  # pylint: disable=protected-access
  return obj
