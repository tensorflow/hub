<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="hub.text_embedding_column" />
<meta itemprop="path" content="Stable" />
</div>

# hub.text_embedding_column

``` python
hub.text_embedding_column(
    key,
    module_spec,
    trainable=False
)
```

Uses a Module to construct a dense representation from a text feature.

This feature column can be used on an input feature whose values are strings
of arbitrary size.

The result of this feature column is the result of passing its `input`
through the module `m` instantiated from `module_spec`, as per
`result = m(input)`. The `result` must have dtype float32 and shape
`[batch_size, num_features]` with a known value of num_features.

Example:

```python
  comment = text_embedding_column("comment", "/tmp/text-module")
  feature_columns = [comment, ...]
  ...
  features = {
    "comment": np.array(["wow, much amazing", "so easy", ...]),
    ...
  }
  labels = np.array([[1], [0], ...])
  input_fn = tf.estimator.inputs.numpy_input_fn(features, labels, shuffle=True)
  estimator = tf.estimator.DNNClassifier(hidden_units, feature_columns)
  estimator.train(input_fn, max_steps=100)
```

#### Args:

* <b>`key`</b>: A string or `_FeatureColumn` identifying the text feature.
* <b>`module_spec`</b>: A ModuleSpec defining the Module to instantiate or a path where
    to load a ModuleSpec via `load_module_spec`
* <b>`trainable`</b>: Whether or not the Module is trainable. False by default,
    meaning the pre-trained weights are frozen. This is different from the
    ordinary tf.feature_column.embedding_column(), but that one is intended
    for training from scratch.


#### Returns:

`_DenseColumn` that converts from text input.


#### Raises:

* <b>`ValueError`</b>: if module_spec is not suitable for use in this feature column.