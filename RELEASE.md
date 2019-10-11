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

# Current version 0.7.0-dev
  * Documentation update for TensorFlow 2.
      * Use `hub.load()` and `hub.KerasLayer` with TF2 (also works in 1.15).
      * For TF1, `hub.Module` and is associated APIs remain available.

# Release 0.6.0
  * Added two examples for exporting of Hub/TF2 modules.
  * Switched to `dense_features_v2` if a high enough TF version is installed.
  * Added `tools/make_image_classifier` for use with TF2.

# Release 0.5.0
  * Fixes in `hub.KerasLayer` related with regularizers, config serialization
    and usage with `tf.estimator`.
  * Updates to feature columns to be compatible with feature columns V2 apis and
    add `hub.sparse_text_embedding_column` for bag of words features.
  * Made `hub.Module` usable within `tf.compat.v1.wrap_function`.

# Release 0.4.0
  * `hub.KerasLayer` (for Hub/TF2 modules) can be used in graph mode
     and can be converted to/from a Keras config.
  * In TF1.x, `hub.Module` can be used inside a defun (helps with TPU).
  * References to TensorFlow Estimator correctly pick up its v1 API.

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
