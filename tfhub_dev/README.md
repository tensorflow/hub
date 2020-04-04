# External contributions to tfhub.dev

## Overview
This location serves for storing documentation for models to be published on
[tfhub.dev](https://tfhub.dev).

Please note that publishing to tfhub.dev is in **early testing**. If you are
interested in publishing, please follow the directions below and we will process
your submission as soon as possible.

### Handles
Model handles are alphanumeric tokens separated by slashes into logical parts.

See example below:

| Handle | Url |
| ------ | --- |
| vtab/jigsaw/1 | https://tfhub.dev/vtab/jigsaw/1 |

### Model storage
Actual models are stored in a
[GCS bucket](https://cloud.google.com/storage/docs/introduction). The tfhub.dev
server distinguishes requests for model documentation from requests for models
stored in GCS bucket via a query parameter:

| Handle | Url for model download |
| ------ | --- |
| vtab/jigsaw/1 | https://tfhub.dev/vtab/jigsaw/1?tf-hub-format=compressed |

More information about the hosting protocol [here](../docs/hosting.md).

## How to publish

The full process of publishing consists of:

1. Creating the model,
1. Packaging the model,
1. Writing documentation,
1. Creating a publishing request.

See sections below for more details.

### Model

<details>
<summary>How to create/export the model</summary>
### Exporting a model

The tfhub.dev repository supports multiple kinds of SavedModel based models,
including:

* [TF2 SavedModels](https://www.tensorflow.org/hub/tf2_saved_model)
  that follow the conventions for [reusable
  SavedModels](https://www.tensorflow.org/hub/reusable_saved_models).
  This is the recommended kind. It supports reusing the SavedModel in a large
  model, including joint training ("fine-tuning").
* SavedModels for inference.
* The deprecated, TF1-only
  [hub.Module](https://www.tensorflow.org/hub/tf1_hub_module) format.
  This is supported for compatibility with TF1 users and deprecated in
  favor of reusable TF2 SavedModels.

<details>
<summary>Best practices for SavedModel publishing</summary>
#### Best practices for SavedModel publishing

##### Align with an existing model type (if applicable)

*   Are there existing models that address the same task as yours?
    Try to implement the same interface, such that consumers can try out
    different models by simply changing the model name.

*   Common interfaces for some frequently occurring model types are
    described [here](https://www.tensorflow.org/hub/common_signatures).

##### Designing the model interface

*   Write models than can handle an unknown batch size on the first dimension.

*   Avoid non-standard ops. Models are saved to the file system with a
    TensorFlow GraphDef at the core, so consumers will need a version of
    TensorFlow that supports all the ops in them.

*   Think ahead to how a consumer might want to fine-tune your model. Read more
    in the reusable SavedModel
    [guide](https://www.tensorflow.org/hub/reusable_saved_models).
</details>

##### Model directory
After you exported the model, you should get a directory (MODEL_DIR) with:

```
MODEL_DIR
├── saved_model.pb
├── assets
|   └── [tokens.txt]
└── variables
|   └── variables@1
└── [tfhub_module.pb]   -> Only for deprecated TF1 Hub modules.
```
</details>

<details>
<summary>How to package the model</summary>

### Packaging the model

The tfhub.dev repository serves compressed SavedModels to save network traffic
and the TF-Hub library supports loading of such compressed models under certain
conditions:

* Model is compressed as tar.gz.
* Model files (like saved_model.pb) and directories are stored at the **root**
  of the archive.

This can be achieved using the following command:

```bash
MODEL_DIR=...       # Directory of my model. This contains at the minimum file
                    # "saved_model.pb", and dirs "assets" and "variables".

tar -cvz -f my_model.tar.gz --owner=0 --group=0 -C "${MODEL_DIR}" .  # Don't leave out the last dot.
```

</details>

### Documentation

<details>
<summary>How to write model documentation</summary>

#### Writing the documentation

The model documentation is a markdown file with a light add-on syntax. See
example below for a minimal example or
[a more realistic example markdown file](examples/example-markdown.md).

<details>
<summary>Example documentation</summary>

```markdown
# Module google/text-embedding-model/1
Simple one sentence description.

<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- module-type: text-embedding -->
<!-- fine-tunable: true -->
<!-- format: saved_model_2 -->

## Overview
Here we give more information about the model.
```
</details>

Currently the markdown file is expected to be structured in the following way:

1. First line is in the form `# Module publisher/model-name/version`
1. Second line starts a one sentence **description** that can span multiple
   lines.
1. Next follows a metadata section that is encoded as key-value pairs inside
   HTML comments. Following are the currently **required metadata** values:

   * `asset-path`: Where should the model be downloaded from at mirroring time.
     This must be a tar.gz'd model as documented in the
     [packaging](#packaging-the-model) section.
   * `module-type`: What is the problem domain of this model. This has to start
     with any of the following: `image-`, `text-`, `audio-`, `video-`. In
     general, any suffix is accepted, but some good (literal) examples are:
       * `text-embedding`
       * `text-generation`
       * `image-augmentation`
       * `image-classification`
       * `image-feature-vector`
       * `image-generator`
       * `image-object-detection`
       * `image-segmentation`
       * `audio-pitch-extraction`

     See problem domains at [tfhub.dev](https://tfhub.dev/s) for more
     inspiration.
   * `fine-tunable`: Can this model be fine-tuned: `[true|false]`.
   * `format`: What is the format of the model:
     `[saved_model_2|saved_model|hub_module]`. See
     [TF2 SavedModel guide](https://www.tensorflow.org/hub/tf2_saved_model)
     for more information.
1. Next follows a free-form markdown section starting with any markdown heading.

Where does the markdown file go?

1. It has to be submitted inside the publisher directory,
   e.g. `.../assets/publisher/...`.
2. It has to end with `.md`.

Why is the metadata required?

> Adding a tiny bit of structure makes it easier/more likely users will find and
> use the assets you've published. Metadata enables filtering, searching and
> ranking to return the most relevant model for a particular user, and improves
> the overall user experience and product quality of tfhub.dev.

<details>
<summary>License</summary>

The default assumed license for a published model is
[Apache 2.0 License](https://opensource.org/licenses/Apache-2.0). The other
accepted options for license are listed in
[OSI Approved Licenses](https://opensource.org/licenses). The possible (literal)
values are:

  * `Apache-2.0`
  * `BSD-3-Clause`
  * `BSD-2-Clause`
  * `GPL-2.0`
  * `GPL-3.0`
  * `LGPL-2.0`
  * `LGPL-2.1`
  * `LGPL-3.0`
  * `MIT`
  * `MPL-2.0`
  * `CDDL-1.0`
  * `EPL-2.0`
  * `custom` - a custom license will require special consideration case by
   case.

An example metadata line with a license other than Apache 2.0:

```markdown
<!-- license: BSD-3-Clause -->
```

</details>

</details>


### Submission

<details>
<summary>How to submit the model</summary>

#### Submitting the model

After the right location of the markdown file is identified (see the
[Writing the documentation](#writing-the-documentation) section above),
the file can be pulled into the master branch of
[tensorflow/hub](https://github.com/tensorflow/hub/tree/master/tensorflow_hub)
by one of the following methods.

##### Git CLI submission

Assuming the identified markdown file path is
`tfhub_dev/assets/publisher/model/1.md`, you can follow the standard Git[Hub]
steps to create a new Pull Request with a newly added file.

This starts with forking the TensorFlow Hub GitHub repository, then creating a
[Pull Request from this fork](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork)
into TensorFlow Hub master branch.

The following are typical CLI git commands needed to adding a new file to a
master branch of the forked repository.

```bash
git clone https://github.com/[github_username]/hub.git
cd hub
mkdir -p tfhub_dev/assets/publisher/model
cp my_markdown_file.md ./tfhub_dev/assets/publisher/model/1.md
git add *
git commit -m "Added model file."
git push origin master
```

##### GitHub GUI submission

A somewhat more straightforward way of submitting is via GitHub graphical user
interface. GitHub allows creating PRs for
[new files](https://help.github.com/en/github/managing-files-in-a-repository/creating-new-files)
or
[file edits](https://help.github.com/en/github/managing-files-in-a-repository/editing-files-in-your-repository)
directly through GUI.

1. On the [TensorFlow Hub GitHub page](https://github.com/tensorflow/hub),
   press `Create new file` button.
1. Set the right file path: `hub/tfhub_dev/assets/publisher/model/1.md`
1. Copy-paste the existing markdown.
1. At the bottom, select "Create a new branch for this commit and start a pull
   request."

</details>


### Validation

<details>
<summary>How to validate model documentation</summary>

#### Validating the documentation

After the markdown file has been added to a correct location, it can be
validated even before creating a Pull Request. To validate the newly created
documentation markdown file, run from the project root:

```
python tfhub_dev/tools/validator.py
```

This will validate all documentation files, including the one you added and
report any potential issues.

If the validator passes, you can be sure that the presubmit tests will also
pass.
</details>


### Advanced topics

<details>
<summary>Adding publisher information</summary>

#### Publishers
Publisher documentation is declared in the same kind of markdown files, with
slight syntactic differences.

The markdown file for a publisher should be always placed on the following path:
`.../assets/publisher/publisher.md`.

See the minimal publisher documentation example:

```markdown
# Publisher vtab
Visual Task Adaptation Benchmark

[![Icon URL]](https://storage.googleapis.com/vtab/vtab_logo_120.png)

## VTAB
The Visual Task Adaptation Benchmark (VTAB) is a diverse, realistic and
challenging benchmark to evaluate image representations.
```

The example above specifies the publisher name, a short description, path to
icon to use, and a longer free-form markdown documentation.

The publisher markdown file is validated the same way as models, see validation
section above.

</details>

<details>
<summary>Creating collections</summary>

#### Collections
Collections are a feature of tfhub.dev that enables publishers to bundle related
models together to improve user search experience.

See the [list of all collections](https://tfhub.dev/s?subtype=model-family) on
tfhub.dev.

See the minimal publisher documentation example:

```markdown
# Collection vtab/benchmark/1
Collection of visual representations that have been evaluated on the VTAB
benchmark.

<!-- module-type: image-feature-vector -->

## Overview
This is the list of visual representations in TensorFlow Hub that have been
evaluated on VTAB. Results can be seen in
[google-research.github.io/task_adaptation/](https://google-research.github.io/task_adaptation/)

#### Models
|                   |
|-------------------|
| [vtab/sup-100/1](https://tfhub.dev/vtab/sup-100/1)   |
| [vtab/rotation/1](https://tfhub.dev/vtab/rotation/1) |
|------------------------------------------------------|
```

The example specifies the name of the collection, a short one sentence
description, problem domain metadata and free-form markdown documentation.

The collection markdown file is validated the same way as models, see validation
section above.

</details>


<details>
<summary>How to organize markdown files</summary>

#### Content organization
The models are stored in the `assets` directory, which is organized into
publisher top level directories.

A Publisher may choose to organize their assets (models, collections) in the
following way:

```
assets
├── publisher_1
│   ├── publisher_1.md     -> Documentation of the publisher.
│   └── models
│       └── model          -> Model name with slashes encoded as sub-path.
│           ├── 1.md       -> Documentation of the model version 1.
│           └── 2.md       -> Documentation of the model version 2.
├── publisher_2
│   ├── publisher_2.md     -> Documentation of the publisher.
│   ├── models
│   │   └── ...
│   └── collections
│       └── collection     -> Documentation for the collection feature.
│           └── 1.md
├── publisher_3
│   └── ...
└── ...
```

</details>
