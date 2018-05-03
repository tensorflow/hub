# TensorFlow Hub

## Introduction

TensorFlow Hub is a library to foster the publication, discovery, and
consumption of reusable parts of machine learning models. A **module** is a
self-contained piece of a TensorFlow graph, along with its weights and assets,
that can be reused across different tasks in a process known as *transfer
learning*.

Modules contain variables that have been pre-trained for a task using a large
dataset. By reusing a module on a related task, you can:

*   **train a model with a smaller dataset**,
*   **improve generalization**, or
*   **significantly speed up training**.

Here's an example that uses an English embedding module to map an array of
strings to their embeddings:

```python
import tensorflow as tf
import tensorflow_hub as hub

with tf.Graph().as_default():
  embed = hub.Module("https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1")
  embeddings = embed(["A long sentence.", "single-word", "http://example.com"])

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    print(sess.run(embeddings))
```


## Getting Started

*   [Installation](installation.md)
*   Tutorials:
    *   [Image Retraining](tutorials/image_retraining.md)
    *   [Text Classification](tutorials/text_classification.md)
    *   [Additional Examples](https://github.com/tensorflow/hub/tree/r0.1/examples/)
*   Key Concepts:
    *   [Using a Module](basics.md)
    *   [Creating a New Module](creating.md)
    *   [Fine-Tuning a Module](fine_tuning.md)
    *   [Hosting a Module](hosting.md)
*   Modules:
    *   [Available Modules](modules/index.md) -- quick links:
        [image](modules/image.md), [text](modules/text.md),
        [other](modules/other.md)
    *   [Common Signatures for Modules](common_signatures/index.md)


## Additional Information

### Publishing on tfhub.dev

Are you interested in publishing your modules on tfhub.dev? Express your
interest via our [Publisher
Survey](https://services.google.com/fb/forms/tensorflowhubpublishersurvey/). We
appreciate your valuable feedback, and will be providing more information about
publishing modules in the coming months. For now, please read our documentation
about [Hosting a Module](hosting.md).

### Fairness

As in all of machine learning, [fairness](http://ml-fairness.com) is an
[important](https://research.googleblog.com/2016/10/equality-of-opportunity-in-machine.html)
consideration. Modules typically leverage large pretrained datasets. When
reusing such a dataset, it’s important to be mindful of what data it contains
(and whether there are any existing biases there), and how these might impact
your downstream experiments.


### Status

Although we hope to prevent breaking changes, this project is still under active
development and is not yet guaranteed to have a stable API or module format.


### Security

Since they contain arbitrary TensorFlow graphs, modules can be thought of as
programs. [Using TensorFlow Securely](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md)
describes the security implications of referencing a module from an untrusted
source.


### Source-Code & Bug Reports

The source code is available on [GitHub](https://github.com/tensorflow/hub).
Use [GitHub issues](https://github.com/tensorflow/hub/issues) for feature requests
and bugs. Please see the [TensorFlow Hub mailing
list](https://groups.google.com/a/tensorflow.org/forum/#!forum/hub) for general
questions and discussion.
