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
"""ModuleImpl interface.

In order to be able to expand the types of Modules that are supported without
users having to call the right constructor we use a "pointer-to-implementation"
pattern:

`Module` is the public API class that every user should instantiate. It's
constructor uses `spec` to create a `ModuleImpl` that encapsulates each specific
implementation.
"""

import abc


class ModuleImpl(object):
  """Internal module implementation interface."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def create_apply_graph(self, signature, input_tensors, name):
    """Applies the module signature to inputs.

    Args:
      signature: A string with the signature to create.
      input_tensors: A dictionary of tensors with the inputs.
      name: A name scope under which to instantiate the signature.

    Returns:
      A dictionary of output tensors from applying the signature.
    """
    raise NotImplementedError()

  def get_signature_name(self, signature):
    """Resolves a signature name."""
    if not signature:
      return "default"
    return signature

  @abc.abstractmethod
  def export(self, path, session):
    """See `Module.export()`."""
    raise NotImplementedError()

  @abc.abstractproperty
  def variable_map(self):
    """See `Module.variable_map`."""
    raise NotImplementedError()
