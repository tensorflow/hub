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
"""Internal. Registry holds python objects that can be injected."""

from absl import logging


class MultiImplRegister(object):
  """Utility class to inject multiple implementations of methods.

  An implementation must implement __call__ and is_supported with the same
  set of arguments. The registered implementations "is_supported" methods are
  called in reverse order under which they are registered. The first to return
  true is then invoked via __call__ and the result returned.
  """

  def __init__(self, name):
    self._name = name
    self._impls = []

  def add_implementation(self, impl):
    """Register an implementation."""
    self._impls += [impl]

  def __call__(self, *args, **kwargs):
    for impl in reversed(self._impls):
      if impl.is_supported(*args, **kwargs):
        return impl(*args, **kwargs)
      else:
        logging.info("%s %s does not support the provided handle.", self._name,
                     type(impl).__name__)
    raise RuntimeError(
        "Missing implementation that supports: %s(*%r, **%r)" % (
            self._name, args, kwargs))


resolver = MultiImplRegister("resolver")
loader = MultiImplRegister("loader")
