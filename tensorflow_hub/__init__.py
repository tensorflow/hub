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
"""TensorFlow Hub Library."""
# pylint: disable=g-import-not-at-top
# pylint: disable=g-statement-before-imports


# Ensure running under a supported version of Python.
def _ensure_python_version():
  """Raises ImportError if sys.version_info is too old."""
  import sys
  #
  # Update this whenever we need to depend on a newer Python version.
  #
  required_python_version = (3, 5)
  if sys.version_info[0:2] < required_python_version:
    raise ImportError(
        "This version of tensorflow_hub requires Python {required} or newer; "
        "instead detected version {present}".format(
            required=".".join(str(x) for x in required_python_version),
            present=sys.version))

_ensure_python_version()


# Ensure TensorFlow is importable and its version is sufficiently recent. This
# needs to happen before anything else, since the imports below will try to
# import tensorflow, too.
def _ensure_tf_install():
  """Attempt to import tensorflow, and ensure its version is sufficient.

  Raises:
    ImportError: if either tensorflow is not importable or its version is
    inadequate.
  """
  try:
    import tensorflow as tf
  except ImportError:
    # Print more informative error message, then reraise.
    print(
        "\n\nFailed to import tensorflow. Please note that tensorflow is not "
        "installed by default when you install tensorflow_hub. This is so that "
        "users can decide which tensorflow package to use. "
        "To use tensorflow_hub, please install a current version of tensorflow "
        "by following the instructions at https://tensorflow.org/install and "
        "https://tensorflow.org/hub/installation.\n\n")
    raise

  from pkg_resources import parse_version

  #
  # Update this whenever we need to depend on a newer TensorFlow release.
  #
  # NOTE: Put only numeric release versions here, like "1.2.3", and be aware
  # that they will also allow release candidates and even any nightly build
  # starting just after the previous release was cut. That's because
  # pkg_resources.parse_version does not understand 'dev' and 'rc' tags;
  # it just does a lexicgraphic comparison after splitting on dots
  # and character class transitions.
  #
  required_tensorflow_version = "1.15.0"
  if (parse_version(tf.__version__) <
      parse_version(required_tensorflow_version)):
    raise ImportError(
        "\n\nThis version of tensorflow_hub requires tensorflow "
        "version >= {required}; Detected an installation of version {present}. "
        "To proceed, please upgrade tensorflow by following the instructions "
        "at https://tensorflow.org/install and "
        "https://tensorflow.org/hub/installation.\n\n".format(
            required=required_tensorflow_version,
            present=tf.__version__))

_ensure_tf_install()


def _ensure_keras_2_importable():
  """Check that Keras 2 can be used by attempting to import tf_keras.

  Raises:
    ImportError: if using tf.keras would mean using Keras 3 and tf_keras is not
      installed.
  """
  import tensorflow as tf

  version_fn = getattr(tf.keras, "version", None)
  if version_fn and version_fn().startswith("3."):
    try:
      import tf_keras as keras
    except ImportError:
      # Print more informative error message, then reraise.
      print(
          "\n\nFailed to import tf_keras. Please note that tf_keras is not"
          " installed by default when you install tensorflow_hub. This is so"
          " that users can decide which tf_keras package to use. To use"
          " tensorflow_hub, please install a current version of tf_keras.\n\n"
      )
      raise

_ensure_keras_2_importable()


# pylint: disable=g-import-not-at-top
# pylint: disable=g-bad-import-order
# Symbols exposed via tensorflow_hub.
from tensorflow_hub.keras_layer import KerasLayer
from tensorflow_hub.module_v2 import load
from tensorflow_hub.module_v2 import resolve
from tensorflow_hub.version import __version__

from tensorflow_hub.config import _run, _get_extra_deps  # pylint: disable=g-multiple-import
_run()

# pylint: enable=g-bad-import-order
# pylint: enable=g-import-not-at-top

# If __all__ is defined the doc generator script only documents the listed
# objects (__all__ defines which symbols you get with
# `from tensorflow_hub import *`).
__all__ = [
    "KerasLayer",
    "load",
    "resolve",
]

__all__ += _get_extra_deps()
__all__ += []  # End of initialization.
