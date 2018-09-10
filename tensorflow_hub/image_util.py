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
"""Helper functions for TF-Hub modules that handle images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_hub import image_module_info_pb2
from tensorflow_hub import native_module


# hub.Modules for images can provide further information for the utilities
# in this file by attaching an ImageModuleInfo message under this key.
IMAGE_MODULE_INFO_KEY = "image_module_info"


# The externally visible name of the message is hub.ImageModuleInfo
ImageModuleInfo = image_module_info_pb2.ImageModuleInfo  # pylint: disable=invalid-name


def attach_image_module_info(image_module_info):
  """Attaches an ImageModuleInfo message from within a module_fn."""
  native_module.attach_message(IMAGE_MODULE_INFO_KEY, image_module_info)


def get_image_module_info(module_or_spec, required=False):
  """Returns the module's attached ImageModuleInfo message, or None."""
  return module_or_spec.get_attached_message(
      IMAGE_MODULE_INFO_KEY, ImageModuleInfo, required=required)


def get_expected_image_size(module_or_spec, signature=None, input_name=None):
  """Returns expected [height, width] dimensions of an image input.

  Args:
    module_or_spec: a Module or ModuleSpec that accepts image inputs.
    signature: a string with the key of the signature in question.
      If None, the default signature is used.
    input_name: a string with the input name for images. If None, the
      conventional input name `images` for the default signature is used.

  Returns:
    A list if integers `[height, width]`.

  Raises:
    ValueError: If the size information is missing or malformed.
  """
  # First see if an attached ImageModuleInfo provides this information.
  image_module_info = get_image_module_info(module_or_spec)
  if image_module_info:
    size = image_module_info.default_image_size
    if size.height and size.width:
      return [size.height, size.width]

  # Else inspect the input shape in the module signature.
  if input_name is None:
    input_name = "images"
  input_info_dict = module_or_spec.get_input_info_dict(signature)
  try:
    shape = input_info_dict[input_name].get_shape()
  except KeyError:
    raise ValueError("Module is missing input '%s' in signature '%s'." %
                     (input_name, signature or "default"))
  try:
    _, height, width, _ = shape.as_list()
    if not height or not width:
      raise ValueError
  except ValueError:
    raise ValueError(
        "Shape of module input is %s, "
        "expected [batch_size, height, width, num_channels] "
        "with known height and width." % shape)
  return [height, width]


def get_num_image_channels(module_or_spec, signature=None, input_name=None):
  """Returns expected num_channels dimensions of an image input.

  This is for advanced users only who expect to handle modules with
  image inputs that might not have the 3 usual RGB channels.

  Args:
    module_or_spec: a Module or ModuleSpec that accepts image inputs.
    signature: a string with the key of the signature in question.
      If None, the default signature is used.
    input_name: a string with the input name for images. If None, the
      conventional input name `images` for the default signature is used.

  Returns:
    An integer with the number of input channels to the module.

  Raises:
    ValueError: If the channel information is missing or malformed.
  """
  if input_name is None:
    input_name = "images"
  input_info_dict = module_or_spec.get_input_info_dict(signature)
  try:
    shape = input_info_dict[input_name].get_shape()
  except KeyError:
    raise ValueError("Module is missing input '%s' in signature '%s'." %
                     (input_name, signature or "default"))
  try:
    _, _, _, num_channels = shape.as_list()
    if num_channels is None:
      raise ValueError
  except ValueError:
    raise ValueError(
        "Shape of module input is %s, "
        "expected [batch_size, height, width, num_channels] "
        "with known num_channels" % shape)
  return num_channels
