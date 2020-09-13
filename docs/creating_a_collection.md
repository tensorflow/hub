<!--* freshness: { owner: 'maringeo' reviewed: '2020-09-14' review_interval: '3 months' } *-->

# Creating a collection

Collections are a feature of tfhub.dev that enables publishers to bundle related
models together to improve user search experience.

See the [list of all collections](https://tfhub.dev/s?subtype=model-family) on
tfhub.dev.

The correct location for the collection file on the TensorFlow Hub repo is:
[hub/tfhub_dev/assets/](https://github.com/tensorflow/hub/tree/master/tfhub_dev/assets)/<publisher_name>/<collection_name>/<collection_name.md>

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
