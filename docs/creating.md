# Creating a New Module
## General approach

To define a new module, a publisher calls `hub.create_module_spec()` with a
function `module_fn`. This function constructs a graph representing the module's
internal structure, using `tf.placeholder()` for inputs to be supplied by
the caller. Then it defines signatures by calling
`hub.add_signature(name, inputs, outputs)` one or more times.

For example:

```python
def module_fn():
  inputs = tf.placeholder(dtype=tf.float32, shape=[None, 50])
  layer1 = tf.layers.dense(inputs, 200)
  layer2 = tf.layers.dense(layer1, 100)
  outputs = dict(default=layer2, hidden_activations=layer1)
  # Add default signature.
  hub.add_signature(inputs=inputs, outputs=outputs)

...
spec = hub.create_module_spec(module_fn)
```

The result of `hub.create_module_spec()` can be used, instead of a path,
to instantiate a module object within a particular TensorFlow graph. In
such case, there is no checkpoint, and the module instance will use the
variable initializers instead.

Any module instance can be serialized to disk via its `export(path, session)`
method. Exporting a module serializes its definition together with the current
state of its variables in `session` into the passed path. This can be used
when exporting a module for the first time, as well as when exporting a fine
tuned module.

For compatibility with TensorFlow Estimators, `hub.LatestModuleExporter` exports
modules from the latest checkpoint, just like `tf.estimator.LatestExporter`
exports the entire model from the latest checkpoint.

Module publishers should implement a [common
signature](common_signatures/index.md) when possible, so that consumers can
easily exchange modules and find the best one for their problem.

## Real example
Take a look at our [text embedding module exporter](https://github.com/tensorflow/hub/blob/master/examples/text_embeddings/export.py)
for a real-world example of how to create a module from a common text embedding
format.
