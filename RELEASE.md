<!-- Copyright 2018 The TensorFlow Hub Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================-->

# Release 0.3.0
  * Initial support for Tensorflow 2.0
  * Tensorflow Hub Library API for Tensorflow 2.0
  * Tensorflow Hub Keras API for Tensorflow 2.0
  * Enable using Tensorflow Hub library against Tensorflow 2.0 run-time.

# Release 0.2.0
 * Add support for caching modules on GCS.
 * Add `ModuleSpec.export(export_path, checkpoint_path)` helper for module
   creation from a separately trained checkpoint.
 * Modules now have a key-value store of attached protocol messages
   for domain- or library-specific auxiliary data.
 * Image modules can have input shapes with variable height and width;
   `hub.get_expected_image_size()` comes from an attached ImageModuleInfo
    message.
 * Add `with hub.eval_function_for_module("...") as f: out = f(in)`.
 * New experimental function `hub.create_module_spec_from_saved_model()`.
 * Added property `hub.Module.variables`.
 * Bazel workspace dependency on protobuf library updated to version 3.6.0.
 * Added progress bar for module downloads when interactive.
 * Numerous minor fixes and updates from the master branch.

# Release 0.1.1
 * Removed TensorFlow version checking.

# Release 0.1.0
 * Initial TensorFlow Hub release.

## Requirements
 * tf-nightly>=1.7.0.dev20180308 || tensorflow>=1.7.0rc0
