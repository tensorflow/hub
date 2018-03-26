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
    *   [Additional Colabs](https://github.com/tensorflow/hub/tree/master/examples/colab)
*   Key Concepts:
    *   [Using a Module](basics.md)
    *   [Creating a New Module](creating.md)
    *   [Fine-Tuning a Module](fine_tuning.md)
*   Modules:
    *   [Available Modules](modules/index.md) -- quick links:
        [image](modules/image.md), [text](modules/text.md),
        [other](modules/other.md)
    *   [Common Signatures for Modules](common_signatures/index.md)


## Additional Information

### Fairness

Please see Google's new website on [ML Fairness](http://ml-fairness.com),
describing potential fairness-related issues in pre-trained modules, including
unintended biases.


### Status

Although we hope to prevent breaking changes, this project is still under active
development and is not yet guaranteed to have a stable API or module format.


### Security

Since they contain arbitrary TensorFlow graphs, modules can be thought of as
programs. [Using TensorFlow Securely](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md)
describes the security implications of referencing a module from an untrusted
source.
