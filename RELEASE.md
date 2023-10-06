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

# Current version 0.15.0
  * Require Python 3.9+.
  * Bump rules_license to 0.0.4.
  * Move `LICENSE` to `tensorflow_hub/LICENSE`.
  * Use `license_files` instead of `license_file` in `setup.cfg`.
  * Remove empty `tools` directory.

# Current version 0.14.0
  * Remove make_image_classifier and make_nearest_neighbour_index.
  * Directly load https://hub.tensorflow.google.cn/* handles from GCS.

# Release 0.13.0
  * Minor fixes for importing Keras and TF Estimator.
  * Stop using distutils.version.
  * Require protobuf >= 3.19.6.
  * Require Python 3.7+.

# Release 0.12.0
  * Improve support for `compute_output_shape` in `hub.KerasLayer`:
    * This will now use the `output_shape` value provided on init, if present,
      before falling back to the default behavior.
  * Changes to make_image_classifier_tool:
    * Add the option to read input with a tf.data.Dataset and use TF ops for
      preprocessing. Enabling this feature requires TF 2.5 or higher.
    * Set the default value of shear_range to 0. shear_range is deprecated and
      incompatible when using TF ops for preprocessing.

# Release 0.11.0
  * Use the Keras load context in keras_layer.py.
  * Always use compressed model loading by default.
  * Documentation improvements.

# Release 0.10.0
  * Enforce Python 3.5+ and TF1.15+.
  * Add ModelLoadFormat setting to switch between (un)compressed model loading.
  * Support for RaggedTensor inputs/outputs is backported from TF2 SavedModels
     to the deprecated hub.Module class (Use of tf.compat.v1.ragged.placeholder()
     in a module_fn is broken for TF2.3 and requires TF2.2 or TF2.4).
  * Bug fixes.

# Release 0.9.0
  * Add SavedModel.LoadOptions to hub.KerasLayer API to pass to load_v2.
  * Improved error messaging.
  * Documentation improvements.
  * Bug fixes.

# Release 0.8.0
  * Implemented make_nearest_neighbour_index tool.
  * Added text FeatureColumn, hub.text_embedding_column_v2, for TF 2.x.
  * CORD-19 embedding colab.
  * Documentation improvements.
  * Bug fixes.

# Release 0.7.0
  * Added support for HubModule v1 in KerasLayer with default tags/signatures.
  * Added support for the KerasLayer to specify tags, signature, as_dict, and
    output_key arguments.
  * Miscellaneous fixes to `hub.KerasLayer`
  * Documentation update for TensorFlow 2.
      * Use `hub.load()` and `hub.KerasLayer` with TF2 (also works in 1.15).
      * For TF1, `hub.Module` and it's associated APIs remain available.

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
