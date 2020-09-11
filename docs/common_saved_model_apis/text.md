<!--* freshness: { owner: 'arnoegw' reviewed: '2020-09-11' } *-->

# Common SavedModel APIs for Text Tasks

This page describes how [TF2 SavedModels](../tf2_saved_model.md) for
text-related tasks should implement the
[Reusable SavedModel API](../reusable_saved_models.md). (This replaces the
[Common Signatures for Text](../common_signatures/text.md) for the
now-deprecated [TF1 Hub format](../tf1_hub_module).)

<a name="feature-vector"></a>

## Text feature vector

A **text feature vector** model creates a dense vector representation from text
features. It accepts a batch of strings of shape `[batch_size]` and maps them to
a `float32` tensor of shape `[batch_size, N]`. This is often called **text
embedding** in dimension `N`.

### Usage summary

```python
obj = hub.load("path/to/model")  # That's tf.saved_model.load() after download.
sentences = ["A long sentence.",
             "single-word",
             "http://example.com"]
representations = obj(sentences)
```

In Keras, the equivalent is

```python
representations = hub.KerasLayer("path/to/model")(sentences)
```

### API details

The [Reusable SavedModel API](../reusable_saved_models.md) also provides a list
of `obj.variables` (e.g., for initialization when not loading eagerly).

A model that supports fine-tuning provides a list of `obj.trainable_variables`.
It may require you to pass `training=True` to execute in training mode (e.g.,
for dropout). The model may also provide a list of `obj.regularization_losses`.
For details, see the [Reusable SavedModel API](../reusable_saved_models.md).

In Keras, this is taken care of by `hub.KerasLayer`: initialize it with
`trainable=True` to enable fine-tuning.

### Notes

Models have been pre-trained on different domains and/or tasks, and therefore
not every text feature vector model would be suitable for your problem. In
particular, some models are trained on a single language.

This interface does not allow fine-tuning of the text representation on TPUs,
because it does not accommodate a split between the trainable variables (on TPU)
and string processing (on CPU).

### Examples

Reusable SavedModels for text feature vectors are used in the Colab tutorial
[Text Classification with Movie Reviews](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb)
