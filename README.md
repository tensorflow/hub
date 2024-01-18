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

<!--
This file is rendered on github.com/tensorflow/hub.
g3doc/_index.yaml is rendered on tensorflow.org/hub.
Both link to g3doc/overview.md and g3doc/*.md for detailed docs.
-->

**Warning: unmigrated tfhub.dev model artifacts will be deleted on March 18,
2024.**

As of November 15th 2023, most [tfhub.dev](https://tfhub.dev) URLs and model
handles are now redirecting to their migrated/equivalent counterpart on Kaggle
Models.

On March 18, 2024, all unmigrated model assets previously surfaced on tfhub.dev
will be deleted – after this date, `hub.load` and `hub.KerasLayer` calls to
these tfhub.dev handles will fail permanently. See the list of unmigrated model
assets here:

-   [inaturalist/vision/embedder/inaturalist_V2](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/inaturalist/models/vision/embedder/inaturalist_V2)
-   [nvidia/unet/industrial/class_1](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/nvidia/models/unet/industrial/class_1)
-   [nvidia/unet/industrial/class_2](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/nvidia/models/unet/industrial/class_2)
-   [nvidia/unet/industrial/class_3](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/nvidia/models/unet/industrial/class_3)
-   [nvidia/unet/industrial/class_4](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/nvidia/models/unet/industrial/class_4)
-   [nvidia/unet/industrial/class_5](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/nvidia/models/unet/industrial/class_5)
-   [nvidia/unet/industrial/class_6](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/nvidia/models/unet/industrial/class_6)
-   [nvidia/unet/industrial/class_7](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/nvidia/models/unet/industrial/class_7)
-   [nvidia/unet/industrial/class_8](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/nvidia/models/unet/industrial/class_8)
-   [nvidia/unet/industrial/class_9](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/nvidia/models/unet/industrial/class_9)
-   [nvidia/unet/industrial/class_10](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/nvidia/models/unet/industrial/class_10)
-   [silero/silero-stt/de](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/silero/models/silero-stt/de)
-   [silero/silero-stt/en](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/silero/models/silero-stt/en)
-   [silero/silero-stt/es](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/silero/models/silero-stt/es)
-   [svampeatlas/vision/classifier/fungi_mobile_V1](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/svampeatlas/models/vision/classifier/fungi_mobile_V1)
-   [svampeatlas/vision/embedder/fungi_V2](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/svampeatlas/models/vision/embedder/fungi_V2)

**If you are an owner of an unmigrated model, please get in touch with us at
kaggle-models@google.com if you'd like to migrate your model. If you take no
action, your model(s) will be deleted on March 18, 2024 and not retrievable
(either by you or other users).**

For models with a Kaggle Models copy, there will be no impact on the
availability/functionality of models that were copied from tfhub.dev –
`tensorflow_hub` will continue to support downloading models that were initially
uploaded to tfhub.dev via
e.g. `hub.load("https://tfhub.dev/<publisher>/<model>/<version>")`. To see if a
tfhub.dev model has been migrated, enter the model handle in your URL bar – if
the redirect is successful, it has already been migrated, otherwise it is an
unmigrated model and will be subject to deletion.

Although no migration or code rewrites are explicitly required, we recommend
replacing tfhub.dev links with their Kaggle Models counterparts to improve code
health and debuggability.

See FAQs [here](https://kaggle.com/tfhub-dev-faqs).

# tensorflow_hub

This GitHub repository hosts the `tensorflow_hub` Python library to download
and reuse SavedModels in your TensorFlow program with a minimum amount of code,
as well as other associated code and documentation.

## Getting Started

  * [Introduction](https://www.tensorflow.org/hub/)
  * The asset types of [tfhub.dev](https://tfhub.dev/)
      * [SavedModels for TensorFlow 2](docs/tf2_saved_model.md)
        and the [Reusable SavedModel interface](docs/reusable_saved_models.md).
      * Deprecated: [Models in TF1 Hub format](docs/tf1_hub_module.md) and
        their [Common Signatures](docs/common_signatures/index.md) collection.
  * Using the library
      * [Installation](docs/installation.md)
      * [Caching model downloads](docs/caching.md)
      * [Migration to TF2](docs/migration_tf2.md)
      * [Model compatibility for TF1/TF2](docs/model_compatibility.md)
      * [Common issues](docs/common_issues.md)
      * [Build from source](docs/build_from_source.md)
      * [Hosting a module](docs/hosting.md)
  * Tutorials
      * [TF2 Image Retraining](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb)
      * [TF2 Text Classification](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb)
      * [Additional TF1 and TF2 examples](examples/README.md)


## Contributing

If you'd like to contribute to TensorFlow Hub, be sure to review the
[contribution guidelines](CONTRIBUTING.md). To contribute code to the
library itself (not examples), you will probably need to
[build from source](docs/build_from_source.md).

This project adheres to TensorFlow's
[code of conduct](https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

We use [GitHub issues](https://github.com/tensorflow/hub/issues) for tracking
requests and bugs.


## License

[Apache License 2.0](LICENSE)
