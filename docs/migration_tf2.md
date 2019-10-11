# Migrating from TF1 to TF2 with TensorFlow Hub

This page explains how to keep using TensorFlow Hub while
migrating your TensorFlow code from TensorFlow 1 to TensorFlow 2.
It complements TensorFlow's general
[migration guide](https://www.tensorflow.org/guide/migrate).

For TF2, TF Hub has switched away from the
[custom hub.Module format](tf1_hub_module.md) and its `hub.Module` API
to the native [SavedModel format of TF2](tf2_saved_model.md)
and its associated API of `hub.load()` and `hub.KerasLayer`.

The `hub.Module` API remains available in the `tensorflow_hub` library
for use in TF1 and in the TF1 compatibility mode of TF2.
It can only load assets in the hub.Module format.

The new API of `hub.load()` (and `hub.KerasLayer`, which wraps it for Keras)
works for TensorFlow 1.15 (in eager and graph mode) and in TensorFlow 2.
This new API can load the new TF2 SavedModel assets, and, with
the restrictions laid out below, for the older hub.Module assets.

In general, it is recommended to use new API wherever possible.

## Summary of the new API

`hub.load()` is the new low-level function to load a SavedModel from
TensorFlow Hub (or compatible services). It wraps TF2's `tf.saved_model.load()`;
TensorFlow's [SavedModel Guide](https://www.tensorflow.org/guide/saved_model)
describes what you can do with the result.

```python
m = hub.load(handle)
outputs = m(inputs)
```

The `hub.KerasLayer` class calls `hub.load()` and adapts the result for
use in Keras alongside other Keras layers. (It may even be a convenient
wrapper for loaded SavedModels used in other ways.)

```python
model = tf.keras.Sequential([
    hub.KerasLayer(handle),
    ...])
```

Hub's tutorials are being updated to the new APIs. See in particular

  * [Text classification example notebook](https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb)
  * [Image classification example notebook](https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb)

If the hub.Module you use has a newer version that comes in the TF2 SavedModel
format, we recommend to switch the API and the module version at the same time.

## Loading old hub.Modules

It can happen that a new TF2 SavedModel is not yet available for your
use-case and you need to load an old hub.Module.

If you use Keras, please wait for `tensorflow_hub` release 0.7, which will
support this in `hub.KerasLayer` soon.

As of this writing, only `hub.load()` supports loading of the hub.Modules
of TF1 into a TF2 program. The code is similar to calling a serving signature.

Instead of

```python
# DEPRECATED: TensorFlow 1
m = hub.Module(handle)
tensor_out = m(tensor_in)
with tf.train.SingularMonitoredSession() as sess:
  print(sess.run(tensor_out))
```

you can write

```python
# TensorFlow 2
m = hub.load(handle, tags=[])
tensors_out_dict = m.signatures["default"](tensor_in)
tensor_out = tensors_out_dict["default"]
print(tensor_out.numpy())  # If executing in eager mode.
```

by spelling out the default tag set, signature name and output tensor key
of a hub.Module.

More generally, instead of

```python
# DEPRECATED: TensorFlow 1
m = hub.Module(handle, tags={"foo", "bar"})
tensors_out_dict = m(dict(x1=..., x2=...), signature="sig", as_dict=True)
```

you can write

```python
# TensorFlow 2
m = hub.load(handle, tags={"foo", "bar"})
tensors_out_dict = m.signatures["sig"](x1=..., x2=...)
```

In these examples `m.signatures` is a dict of TensorFlow [concrete
functions](https://www.tensorflow.org/tutorials/customization/performance#tracing)
keyed by signature names. Calling such a function computes all its outputs,
even if unused. (This is different from the lazy evaluation of TF1's
graph mode.)

Retraining hub.Modules loaded via `hub.load()` is not supported:
Trainable variables are imported as such, but update ops (for batch
normalization etc.) and regularization losses are dropped.
Be sure to *not* capture the trainable variables of `m` in a gradient tape
or otherwise in an optimizer. Do *not* import `tags={"train"}`.
