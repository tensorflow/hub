# Modules

**TensorFlow Hub** is a new library for creating reusable pieces of machine
learning models, known as *modules*. A module is a self-contained piece of a
TensorFlow graph, along with its weights and assets, that can be transferred
from one task to another.

Learn how to use TensorFlow Hub by following the [image
retraining](https://www.tensorflow.org/tutorials/image_retraining) tutorial,
trying our [text classification](TODO) Colab, or browsing our [source
code](https://github.com/tensorflow/hub).

Pre-trained TensorFlow Hub modules are already available for a variety of
tasks. They are described in the following documents:

  * [Image Modules](image.md), which lists modules trained extensively to
    classify objects in images. By reusing their feature detection capabilities,
    you can create models that recognize your own classes using much less
    training data and time.
  * [Text Modules](text.md), which lists modules with pre-trained text
    embeddings that can be used to classify text. These can be simple lookup
    tables or more complicated designs, usually accepting full sentences or
    paragraphs.
  * [Other Modules](other.md), which lists modules for other types of tasks,
    such as mapping from latent space to images or extracting deep local
    features.

Please see Google's new website on [ML Fairness](http://ml-fairness.com),
describing potential fairness-related issues in pre-trained modules, including
unintended biases.
