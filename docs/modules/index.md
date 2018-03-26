# Modules

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

As in all of machine learning, [fairness](http://ml-fairness.com) is an
[important](https://research.googleblog.com/2016/10/equality-of-opportunity-in-machine.html)
consideration. Modules typically leverage large pretrained datasets. When
reusing such a dataset, it’s important to be mindful of what data it contains
(and whether there are any existing biases there), and how these might impact
your downstream experiments.
