<!--* freshness: { owner: 'maringeo' reviewed: '2020-09-14' review_interval: '3 months' } *-->

# Contribute a model

## Submitting the model

After the right location of the markdown file is identified (see the
[writing model documentation](writing_model_documentation.md) guide), the file
can be pulled into the master branch of
[tensorflow/hub](https://github.com/tensorflow/hub/tree/master/tensorflow_hub)
by one of the following methods.

### Git CLI submission

Assuming the identified markdown file path is
`tfhub_dev/assets/publisher/model/1.md`, you can follow the standard Git[Hub]
steps to create a new Pull Request with a newly added file.

This starts with forking the TensorFlow Hub GitHub repository, then creating a
[Pull Request from this fork](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork)
into the TensorFlow Hub master branch.

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

### GitHub GUI submission

A somewhat more straightforward way of submitting is via GitHub graphical user
interface. GitHub allows creating PRs for
[new files](https://help.github.com/en/github/managing-files-in-a-repository/creating-new-files)
or
[file edits](https://help.github.com/en/github/managing-files-in-a-repository/editing-files-in-your-repository)
directly through GUI.

1.  On the [TensorFlow Hub GitHub page](https://github.com/tensorflow/hub),
    press `Create new file` button.
1.  Set the right file path: `hub/tfhub_dev/assets/publisher/model/1.md`
1.  Copy-paste the existing markdown.
1.  At the bottom, select "Create a new branch for this commit and start a pull
    request."

## Model page specific markdown format

The model documentation is a markdown file with some add-on syntax. See example
below for a minimal example or
[a more realistic example markdown file](https://github.com/tensorflow/hub/blob/master/tfhub_dev/examples/example-markdown.md).

### Example documentation

A high-quality model documentation contains code snippets, information how the
model was trained and intended usage. You should also make use of model specific
medata properties [below](#model-markdown-specific-metadata-properties) and
general properties described in
[writing model documentation](writing_model_documentation.md).

```markdown
# Module google/text-embedding-model/1
Simple one sentence description.

<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- module-type: text-embedding -->
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

### Model deployments and grouping deployments together.

tfhub.dev allows for publishing TFJS, TFLite and Coral deployments of a
TensorFlow model.

The first line of the markdown file should specify the type of the deployment
format. Use:

*   `# Tfjs publisher/model/version` for TFJS deployments
*   `# Lite publisher/model/version` for Lite deployments
*   `# Coral publisher/model/version` for Coral deployments

It it a good idea for these different deployments to show in the same model page
on tfhub.dev. To associate a given TFJS, TFLite or Coral deployment to a
TensorFlow model, specify the parent-model tag:

```markdown
<!-- parent-model: publisher/model/version -->
```

Sometimes you might want to publish one or more deployments without the
TensorFlow model. In that case, you'll need to create a Placeholder model and
specify its handle in the parent-model tag. The placeholder markdown is
identical to TensorFlow model markdown, except that the first line is: `#
Placeholder publisher/model/version` and it doesn't require the `asset-path`
property.

### Model markdown specific metadata properties

Apart from the shared metadata properties described in
[writing model documentation](writing_model_documentation.md), the model
markdown supports the following properties:

*   `fine-tunable`: whether the model is fine-tunable
*   `format`: the TensorFlow Hub format of the model. Valid values are `hub`
    when the model was exported via the legacy
    [TF1 hub format](exporting_hub_format.md) or `saved_model_2` when the model
    was exportd via a [TF2 Saved Model](exporting_tf2_saved_model.md).
*   `asset-path`: the world-readable remote path to the actual model assets to
    upload, such as on a Google Cloud Storage bucket.
*   `licence`: see section below

### License

The default assumed license for a published model is
[Apache 2.0 License](https://opensource.org/licenses/Apache-2.0). The other
accepted options for license are listed in
[OSI Approved Licenses](https://opensource.org/licenses). The possible (literal)
values are:

*   `Apache-2.0`
*   `BSD-3-Clause`
*   `BSD-2-Clause`
*   `GPL-2.0`
*   `GPL-3.0`
*   `LGPL-2.0`
*   `LGPL-2.1`
*   `LGPL-3.0`
*   `MIT`
*   `MPL-2.0`
*   `CDDL-1.0`
*   `EPL-2.0`
*   `custom` - a custom license will require special consideration case by case.

An example metadata line with a license other than Apache 2.0:

```markdown
<!-- license: BSD-3-Clause -->
```
