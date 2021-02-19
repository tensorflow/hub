<!--* freshness: { owner: 'maringeo' reviewed: '2020-12-30' review_interval: '3 months' } *-->

# Create a collection

Collections are a feature of tfhub.dev that enables publishers to bundle related
models together to improve user search experience.

See the [list of all collections](https://tfhub.dev/s?subtype=model-family) on
tfhub.dev.

The correct location for the collection file in the repository
[github.com/tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev) is
[assets/docs](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs)/<b>&lt;publisher_name&gt;</b>/collections/<b>&lt;collection_name&gt;</b>/<b>1</b>.md

Here is a minimal example that would go into
assets/docs/<b>vtab</b>/collections/<b>benchmark</b>/<b>1</b>.md.
Note how the collection's name in the first line is shorter than the name
of the file.


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
