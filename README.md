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

# TensorFlow Hub

[TensorFlow Hub](https://tfhub.dev) is a repository of reusable assets
for machine learning with [TensorFlow](https://www.tensorflow.org/).
In particular, it provides pre-trained SavedModels that can be reused
to solve new tasks with less training time and less training data.

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

This project adheres to TensorFlow's [code of
conduct](https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md). By
participating, you are expected to uphold this code.

We use [GitHub issues](https://github.com/tensorflow/hub/issues) for tracking
requests and bugs. Please see the [TensorFlow Hub mailing
list](https://groups.google.com/a/tensorflow.org/forum/#!forum/hub) for general
questions and discussion, or tag [tensorflow-hub on Stack
Overflow](https://stackoverflow.com/questions/tagged/tensorflow-hub).


## License

[Apache License 2.0](LICENSE)
