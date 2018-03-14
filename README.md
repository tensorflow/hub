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

# TensorFlow Hub

TensorFlow Hub is a library to foster the publication, discovery, and
consumption of reusable parts of machine learning models. A **module** is a
self-contained piece of a TensorFlow graph, along with its weights and assets,
that can be reused across different tasks.

Typically, modules contain variables that have been pre-trained for a task using
a large dataset. By reusing a module on a related or similar task, a user can
train a model with a smaller dataset, improve generalization, or simply speed up
training.

Modules can be instantiated from a URL or filesystem path while a TensorFlow
graph is being constructed. It can then be *applied* like an ordinary Python
function to build part of the graph. For example:

```python
import tensorflow as tf
import tensorflow_hub as hub

with tf.Graph().as_default():
  # Download a 128-dimension English embedding.
  embed = hub.Module("https://storage.googleapis.com/tensorflow-hub/google/text/nnlm-en-dim128-normalized/1.tar.gz")

  # Use the module to map an array of strings to their embeddings.
  embeddings = embed([
      "A long sentence.",
      "single-word",
      "http://example-url.com"])

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    print(sess.run(embeddings))
```

Each module has a defined interface that allows it to be used in a replaceable
way, with little or no knowledge of its internals. Once exported to disk, a
module is self-contained and can be used by others without access to the code
and data used to create and train it.


## Installation

Currently, TensorFlow Hub depends on bug fixes and enhancements not present in a
stable TensorFlow release. For now, please
[install or upgrade](https://www.tensorflow.org/install/)
TensorFlow package past 1.7.0rc0. For instance:

```bash
pip install --upgrade tensorflow>=1.7.0rc0
pip install --upgrade tensorflow-hub
```

This section will be updated to include a specific TensorFlow version
requirement when a compatible release is made available.


## Status

Although we hope to prevent breaking changes, this project is still under active
development and is not yet guaranteed to have a stable API or module format.


## Security

Since they contain arbitrary TensorFlow graphs, modules can be thought of as
programs. [Using TensorFlow Securely](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md)
describes the security implications of referencing a module from an untrusted
source.


## Key Concepts

### Instantiating a Module

A TensorFlow Hub module is imported into a TensorFlow program by
creating a `Module` object from a string with its URL or filesystem path,
such as:

```python
m = hub.Module("path/to/a/module_dir")
```

This adds the module's variables to the current TensorFlow graph.
Running their initializers will read their pre-trained values from disk.
Likewise, tables and other state is added to the graph.

#### Caching Modules

When creating a module from a URL, the module content is downloaded
and cached in the local system temporary directory. The location where
modules are cached can be overridden using `TFHUB_CACHE_DIR` environment
variable.

For example, setting `TFHUB_CACHE_DIR` to `/my_module_cache`:

```shell
export TFHUB_CACHE_DIR=/my_module_cache
```

and then creating a module from a URL:

```python
m = hub.Module("https://storage.googleapis.com/tensorflow-hub/google/test/half-plus-two/1.tar.gz")
```

results in downloading the unpacked version of the module in
`/my_module_cache`.


### Applying a Module

Once instantiated, a Module `m` can be called zero or more times
like a Python function from tensor inputs to tensor outputs:

```python
y = m(x)
```

Each such call adds operations to the current TensorFlow graph to compute
`y` from `x`. If this involves variables with trained weights, these are
shared between all applications.

Modules can define multiple named *signatures* in order to allow being applied
in more than one way. A module's documentation should describe the available
signatures. The call above applies the signature named `"default"`. Other
signature names can be specified with the optional `signature=` argument.

If a signature has multiple inputs, they must be passed as a dict,
with the keys defined by the signature. Likewise, if a signature has
multiple outputs, these can be retrieved as a dict by passing `as_dict=True`,
under the keys defined by the signature. (The key `"default"` is for the
single output returned if `as_dict=False`.)
So the most general form of applying a Module looks like:

```python
outputs = m(dict(apples=x1, oranges=x2), signature="my_method", as_dict=True)
y1 = outputs["cats"]
y2 = outputs["dogs"]
```

A caller must supply all inputs defined by a signature, but there is no
requirement to use all of a module's outputs.
TensorFlow will run only those parts of the module that end up
as dependencies of a target in `tf.Session.run()`. Indeed, module publishers may
choose to provide various outputs for advanced uses (like activations of
intermediate layers) along with the main outputs. Module consumers should
handle additional outputs gracefully.

### Creating a New Module

To define a new module, a publisher calls `hub.create_module_spec()` with a
function `module_fn`. This function constructs a graph representing the module's
internal structure, using `tf.placeholder()` for inputs to be supplied by
the caller. Then it defines signatures by calling
`hub.add_signature(name, inputs, outputs)` one or more times.

For example:

```python
def module_fn():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 50])
  layer1 = tf.layers.fully_connected(inputs, 200)
  layer2 = tf.layers.fully_connected(layer1, 100)
  outputs = dict(default=layer2, hidden_activations=layer1)
  # Add default signature.
  hub.add_signature(inputs=x, outputs=outputs)

...
spec = hub.create_module_spec(module_fn)
```

The result of `hub.create_module_spec()` can be used, instead of a path,
to instantiate a module object within a particular TensorFlow graph. In
such case, there is no checkpoint, and the module instance will use the
variables initializers instead.

Any module instance can be serialized to disk via its `export(path, session)`
method. Exporting a module serializes its definition together with the current
state of its variables in `session` into the passed path. This can be used
when exporting a module for the first time, as well as when exporting a fine
tuned module.

Additionally, for compatibility with TensorFlow Estimators, `hub` library
provides a `LatestModuleExporter`.

Module publishers should implement a [common
signature](https://github.com/tensorflow/hub/blob/master/common_signatures/index.md)
when possible, so that consumers can easily exchange modules and find the best
one for their problem.

### Fine Tuning

Training the variables of a consumer model, including those of an imported
module, is called *fine-tuning*. Fine-tuning can result in better quality, but
adds new complications. We advise consumers to look into fine-tuning only after
exploring simpler quality tweaks.

#### For Consumers

To enable fine-tuning, instantiate the module with
`hub.Module(..., trainable=True)` to make its variables trainable and
import TensorFlow's `REGULARIZATION_LOSSES`. If the module has multiple
graph variants, make sure to pick the one approprate for training.
Usually, that's the one with tags `{"train"}`.

Choose a training regime that does not ruin the pre-trained weights,
for example, a lower learning rate than for training from scratch.

#### For Publishers

To make fine-tuning easier for consumers, please be mindful of the following:

*   Fine-tuning needs regularization. Your module is exported with the
    `REGULARIZATION_LOSSES` collection, which is what puts your choice of
    `tf.layers.dense(..., kernel_regularizer=...)` etc. into what the consumer
    gets from `tf.losses.get_regularization_losses()`. Prefer this way of
    defining L1/L2 regularization losses.

*   In the publisher model, avoid defining L1/L2 regularization via the `l1_`
    and `l2_regularization_strength` parameters of `tf.train.FtrlOptimizer`,
    `tf.train.ProximalGradientDescentOptimizer`, and other proximal
    optimizers. These are not exported alongside the module, and setting
    regularization strengths globally may not be appropriate for the
    consumer. Except for L1 regularization in wide (i.e. sparse linear) or wide
    & deep models, it should be possible to use individual regularization losses
    instead.

*   If you use dropout, batch normalization, or similar training techniques, set
    dropout rate and other hyperparameters to values that make sense across many
    expected uses.


## License

[Apache License 2.0](LICENSE)
