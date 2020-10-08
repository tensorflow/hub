<!--* freshness: { owner: 'maringeo' reviewed: '2020-09-14' review_interval: '3 months' } *-->

# Write model documentation

The model documentation that is displayed on tfhub.dev is written in the
Markdown format.

## Types of markdown documentation

There are 3 types of markdown documentation used in tfhub.dev:

*   Publisher markdown - contains information about a publisher (learn more in
    the [become a publisher](publish.md) guide).
*   Model markdown - contains information about a specific model. (learn more in
    the [contribute a model](contribute_a_model.md) guide).
*   Collection markdown - contains information about a publisher-defined
    collection of models. (learn more in the
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

Publisher and collection markdowns are unversioned, while models can have
different versions. Each model version requires a separate markdown file named
after the version it describes (i.e. 1.md, 2.md).

All model versions for a given model should be located in the model directory.

Below is an illustration on how the markdown contant is organized:

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

## Common metadata properties

The collection and model markdowns can contain metadata properties. These are
represented as markdown comments after the first line of the markdown file.

For example, `<!-- network-architecture: BigGAN -->` states that a model or a
collection of models have BigGAN architecture.

The following metadata properties can be used in both model and collection
markdown files:

*   `module-type`: the problem domain, for example "Text embedding" or "Image
    classifier"
*   `dataset`: the dataset the model(s) was trained on.
*   `netowk-architecture`: the network architecture.
*   `language`: commonly used for text models or collections to describe the
    language of the dataset the model was trained on. This property is not
    necessary for non-text models, e.g. an image classifier.
