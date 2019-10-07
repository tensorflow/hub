# Migration of TF-Hub from TensorFlow 1 to TensorFlow 2

With the releases of TensorFlow 2 (TF2) and the object-based SavedModel v2
(SMv2) the recommended way to create and use TF-Hub modules changed. While it is
still possible to use SMv2 in TF1 and SMv1 in TF2, it does not work for all
models and some workflows need to be adapted. The new object-based SMv2 can be
loaded in eager and graph mode in a reusable way. In general, it is always
preferred and recommended to use new TF-Hub API functions.

## Load new object-based SavedModel v2

The API of TF-Hub for TF2 has been completely changed, making previous commands
obsolete. There are now two recommended ways to load a Hub module: `hub.load()`
and `hub.KerasLayer()`.

`hub.load` is the new standard way to load a module:

```python
m = hub.load(handle)
outputs = m(inputs)
```

The `hub.KerasLayer` wrapper allows to make a callable object composed of only
serializable TF primitives into a layer that can be used with other Keras
components. It gives special treatment to the following attributes: `__call__`,
`variables`, `trainable_variables`, and `regularization_losses`.

```python
model = tf.keras.Sequential([
    ...,
    hub.KerasLayer(handle),
    ...])
```

## Load old SavedModel v1

It can happen that a new object-based SavedModel is not yet available for your
use case and you need to load an old SavedModel v1. Since there is no concept of
`tags` in SavedModel v2 anymore, to load an older SMv1 TF-Hub module, the new
API provides additional arguments.

Similar to the old deprecated workflow, tags can be provided at module
instantiation, and the used signature can be specified at the call:

```python
m = hub.load(handle, tags=<list-of-tags>)
outputs = m.signatures[<signature-name>](inputs)
```

Note that if using signatures API, the returned outputs are always dictionaries.

To use an old SMv1 as a keras layer, currently one needs to wrap it in an
checkpointable object:

```python
class HubWrapper(tf.train.Checkpoint):
  def __init__(self, spec, signature="default", tags=None, end_point="default"):
    super(HubWrapper, self).__init__()
    self.module = hub.load(spec, tags=tags)
    self.variables = getattr(self.module, "variables", [])
    self.trainable_variables = getattr(self.module, "trainable_variables", [])
    self.regularization_losses = getattr(self.module, "regularization_losses", [])
    self._signature = signature
    self._end_point = end_point
  def __call__(self, x):
    return self.module.signatures[self._signature](x)[self._end_point]

model = tf.keras.Sequential([
    ...,
    hub.KerasLayer(HubWrapper(handle)),
    ...])
```

## Create a Reusable SavedModel v2

For a general case of a reusable and retrainable object-based SavedModel, the
object is expected to contain `variables`, `trainable_variables`, and
`regularization_losses` attributes. The main single-entry point is the
`__call__()` method, that accepts the inputs, and if needed should be configured
to accept the boolean `training` argument and potential other numeric
hyperparameters.

When exporting a keras model, the recommended way is to use
`tf.keras.models.save_model()` function that automatically attaches additional
objects and functions to the saved model. This is the recommended way when
creating a reusable SMv2 in TF2.

In case a custom model needs to be saved, one can construct a saveable object
manually and to save it via `tf.saved_model.save()` function:

```python
class Object(tf.train.Checkpoint):
  def __init__(self):
    super(Object, self).__init__()
    ...
    self.variables = ...              # List of variables.
    self.trainable_variables = ...    # List of trainable variables.
    self.regularization_losses = ...  # List of regularizer callables that return tensors.

  @tf.function(input_signature=[tf.TensorSpec(shape=<input-shape>, dtype=tf.float32)])
  def __call__(self, inputs):
    return ...

obj = Object()
tf.saved_model.save(obj, export_path)
```

A hub module created in this way will be easily usable as discussed in the
section above.

Note that this is a convention for most common use cases. In some cases, not all
arguments will be needed and in other cases one might want to have a more
customized interface.

## Additional Guides and Examples

*   [TF-2 SavedModel guide](https://www.tensorflow.org/guide/saved_model)
*   [TF-2 Keras SavedModel saving/loading](https://github.com/tensorflow/community/blob/master/rfcs/20190509-keras-saved-model.md)
*   [Text classification example notebook](https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb)
*   [Image module retraining example notebook](https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb)
