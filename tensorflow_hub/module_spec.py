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
"""ModuleSpec interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class ModuleSpec(object):
  """Represents the contents of a Module before it has been instantiated.

  A ModuleSpec is the blueprint used by `Module` to create one or more instances
  of a specific module in one or more graphs. The details on how to construct
  the Module are internal to the library implementation but methods to inspect
  a Module interface are public.

  Note: Do not instantiate this class directly. Use `hub.load_module_spec` or
  `hub.create_module_spec`.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    """Do not instantiate directly."""
    pass

  @abc.abstractmethod
  def get_signature_names(self, tags=None):
    """Returns the module's signature names as an iterable of strings."""
    pass

  @abc.abstractmethod
  def get_tags(self):
    """Lists the graph variants as an iterable of set of tags."""
    return [set()]

  @abc.abstractmethod
  def get_input_info_dict(self, signature=None, tags=None):
    """Describes the inputs required by a signature.

    Args:
      signature: A string with the signature to get inputs information for.
        If None, the default signature is used if defined.
      tags: Optional set of strings, specifying the graph variant to query.

    Returns:
      A dict from input names to objects that provide (1) a property `dtype`,
      (2) a method `get_shape()` and (3) a read-only boolean property
      `is_sparse`. The first two are compatible with the common API of Tensor
      and SparseTensor objects.

    Raises:
      KeyError: if there is no such signature or graph variant.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def get_output_info_dict(self, signature=None, tags=None):
    """Describes the outputs provided by a signature.

    Args:
      signature: A string with the signature to get ouputs information for.
        If None, the default signature is used if defined.
      tags: Optional set of strings, specifying the graph variant to query.

    Returns:
      A dict from input names to objects that provide (1) a property `dtype`,
      (2) a method `get_shape()` and (3) a read-only boolean property
      `is_sparse`. The first two are compatible with the common API of Tensor
      and SparseTensor objects.

    Raises:
      KeyError: if there is no such signature or graph variant.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def _create_impl(self, name, trainable, tags):
    """Internal.

    Args:
      name: A string with the an unused name scope.
      trainable: A boolean, whether the Module is to be instantiated as
        trainable.
      tags: A set of strings specifying the graph variant to use.

    Returns:
      A ModuleImpl.
    """
    raise NotImplementedError()
