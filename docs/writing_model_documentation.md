<!--* freshness: { owner: 'wgierke' reviewed: '2021-05-17' review_interval: '3 months' } *-->

# Write model documentation

For contributing models to tfhub.dev, a documentation in Markdown must be
provided. For a full overview of the process of adding models to tfhub.dev see
the [contribute a model](contribute_a_model.md) guide.

## Types of Markdown documentation

There are 3 types of Markdown documentation used in tfhub.dev:

*   Publisher Markdown - contains information about a publisher (learn more in
    the [become a publisher](publish.md) guide).
*   Model Markdown - contains information about a specific model.
*   Collection Markdown - contains information about a publisher-defined
    collection of models (learn more in the
    [create a collection](creating_a_collection.md) guide).

## Content organization

The following content organization is recommended when contributing to the
[TensorFlow Hub GitHub](https://github.com/tensorflow/hub) repository:

*   each publisher directory is in the `assets` directory.
*   each publisher directory contains optional `models` and `collections`
    directories
*   each model should have its own directory under
    `assets/publisher_name/models`
*   each collection should have its own directory under
    `assets/publisher_name/collections`

Publisher and collection Markdowns are unversioned, while models can have
different versions. Each model version requires a separate Markdown file named
after the version it describes (i.e. 1.md, 2.md).

All model versions for a given model should be located in the model directory.

Below is an illustration on how the Markdown content is organized:

```
assets
├── publisher_name_a
│   ├── publisher_name_a.md  -> Documentation of the publisher.
│   └── models
│       └── model          -> Model name with slashes encoded as sub-path.
│           ├── 1.md       -> Documentation of the model version 1.
│           └── 2.md       -> Documentation of the model version 2.
├── publisher_name_b
│   ├── publisher_name_b.md  -> Documentation of the publisher.
│   ├── models
│   │   └── ...
│   └── collections
│       └── collection     -> Documentation for the collection feature.
│           └── 1.md
├── publisher_name_c
│   └── ...
└── ...
```

## Model page specific Markdown format

The model documentation is a Markdown file with some add-on syntax. See the
example below for a minimal example or
[a more realistic example Markdown file](https://github.com/tensorflow/tfhub.dev/blob/master/examples/docs/tf2_model_example.md).

### Example documentation

A high-quality model documentation contains code snippets, information how the
model was trained and intended usage. You should also make use of model-specific
metadata properties
[explained below](#model-markdown-specific-metadata-properties) so users can
find your models on tfhub.dev faster.

```markdown
# Module google/text-embedding-model/1

Simple one sentence description.

<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- task: text-embedding -->
<!-- fine-tunable: true -->
<!-- format: saved_model_2 -->

## Overview

Here we give more information about the model including how it was trained,
expected use cases, and code snippets demonstrating how to use the model:

``
Code snippet demonstrating use (e.g. for a TF model using the tensorflow_hub library)

import tensorflow_hub as hub

model = hub.KerasLayer(<model name>)
inputs = ...
output = model(inputs)
``
```

### Model deployments and grouping deployments together

tfhub.dev allows publishing TF.js, TFLite and Coral deployments of a TensorFlow
model.

The first line of the Markdown file should specify the type of the deployment
format:

*   `# Tfjs publisher/model/version` for TF.js deployments
*   `# Lite publisher/model/version` for Lite deployments
*   `# Coral publisher/model/version` for Coral deployments

It is a good idea for these different deployments to show up on the same model
page on tfhub.dev. To associate a given TF.js, TFLite or Coral deployment to a
TensorFlow model, specify the parent-model tag:

```markdown
<!-- parent-model: publisher/model/version -->
```

Sometimes you might want to publish one or more deployments without a TensorFlow
SavedModel. In that case, you'll need to create a Placeholder model and specify
its handle in the `parent-model` tag. The placeholder Markdown is identical to
TensorFlow model Markdown, except that the first line is: `# Placeholder
publisher/model/version` and it doesn't require the `asset-path` property.

### Model Markdown specific metadata properties

The Markdown files can contain metadata properties. These are represented as
Markdown comments after the description of the Markdown file, e.g.

```
# Module google/universal-sentence-encoder/1
Encoder of greater-than-word length text trained on a variety of data.

<!-- task: text-embedding -->
...
```

The following metadata properties exist:

*   `format`: For TensorFlow models: the TensorFlow Hub format of the model.
    Valid values are `hub` when the model was exported via the legacy
    [TF1 hub format](exporting_hub_format.md) or `saved_model_2` when the model
    was exported via a [TF2 Saved Model](exporting_tf2_saved_model.md).
*   `asset-path`: the world-readable remote path to the actual model assets to
    upload, such as to a Google Cloud Storage bucket. The URL should be allowed
    to be fetched from by the robots.txt file (for this reason,
    "https://github.com/.*/releases/download/.*" is not supported as it is
    forbidden by https://github.com/robots.txt). See
    [below](#model-specific-asset-content) for more information on the expected
    file type and content.
*   `parent-model`: For TF.js/TFLite/Coral models: handle of the accompanying
    SavedModel/Placeholder
*   `fine-tunable`: Boolean, whether the model can be fine-tuned by the user.
*   `task`: the problem domain, e.g. "text-embedding". All supported values are
    defined in
    [task.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/task.yaml).
*   `dataset`: the dataset the model was trained on, e.g. "wikipedia". All
    supported values are defined in
    [dataset.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/dataset.yaml).
*   `network-architecture`: the network architecture the model is based on, e.g.
    "mobilenet-v3". All supported values are defined in
    [network_architecture.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/network_architecture.yaml).
*   `language`: the language code of the language a text model was trained on,
    e.g. "en". All supported values are defined in
    [language.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/language.yaml).
*   `license`: The license that applies to the model, e.g. "mit". The default
    assumed license for a published model is
    [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0). All
    supported values are defined in
    [license.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/license.yaml).
    Note that the `custom` license will require special consideration case by
    case.
*   `colab`: HTTPS URL to a notebook that demonstrates how the model can be used
    or trained
    ([example](https://colab.sandbox.google.com/github/tensorflow/hub/blob/master/examples/colab/bigbigan_with_tf_hub.ipynb)
    for [bigbigan-resnet50](https://tfhub.dev/deepmind/bigbigan-resnet50/1)).
    Must lead to `colab.research.google.com`. Note that Jupyter notebooks hosted
    on GitHub can be accessed via
    `https://colab.research.google.com/github/ORGANIZATION/PROJECT/
    blob/master/.../my_notebook.ipynb`.
*   `demo`: HTTPS URL to a website that demonstrates how the TF.js model can be
    used ([example](https://teachablemachine.withgoogle.com/train/pose) for
    [posenet](https://tfhub.dev/tensorflow/tfjs-model/posenet/mobilenet/float/075/1/default/1)).

The Markdown documentation types support different required and optional
metadata properties:

| Type        | Required                 | Optional                           |
| ----------- | ------------------------ | ---------------------------------- |
| Publisher   |                          |                                    |
| Collection  | task                     | dataset, language,                 |
:             :                          : network-architecture               :
| Placeholder | task                     | dataset, fine-tunable, language,   |
:             :                          : license, network-architecture      :
| SavedModel  | asset-path, task,        | colab, dataset, language, license, |
:             : fine-tunable, format     : network-architecture               :
| Tfjs        | asset-path, parent-model | colab, demo                        |
| Lite        | asset-path, parent-model | colab                              |
| Coral       | asset-path, parent-model | colab                              |

### Model-specific asset content

Depending on the model type, the following file types and contents are expected:

*   SavedModel: a tar.gz archive containing content like so:

```
saved_model.tar.gz
├── assets/            # Optional.
├── assets.extra/      # Optional.
├── variables/
│     ├── variables.data-?????-of-?????
│     └──  variables.index
├── saved_model.pb
├── keras_metadata.pb  # Optional, only required for Keras models.
└── tfhub_module.pb    # Optional, only required for TF1 models.
```

*   TF.js: a tar.gz archive containing content like so:

```
tf_js_model.tar.gz
├── group*
├── *.json
├── *.txt
└── *.pb
```

*   TFlite: a .tflite file
*   Coral: a .tflite file

Generally, all files and directories (whether compressed or uncompressed) must
start with a word character so e.g. dots are no valid prefix of file
names/directories.
