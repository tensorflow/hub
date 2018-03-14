<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="hub.image_embedding_column" />
</div>

# hub.image_embedding_column

``` python
hub.image_embedding_column(
    key,
    module_spec
)
```

Uses a Module to get a dense 1-D representation from the pixels of images.

This feature column can be used on images, represented as float32 tensors of
RGB pixel data in the range [0,1]. This can be read from a numeric_column()
if the tf.Example input data happens to have decoded images, all with the
same shape [height, width, 3]. More commonly, the input_fn will have code to
explicitly decode images, resize them (possibly after performing data
augmentation such as random crops etc.), and provide a batch of shape
[batch_size, height, width, 3].

The result of this feature column is the result of passing its `input`
through the module `m` instantiated from `module_spec`, as per
`result = m({"images": input})`. The `result` must have dtype float32 and
shape `[batch_size, num_features]` with a known value of num_features.

Example:

```python
  image_column = hub.image_embedding_column("embeddings", "/tmp/image-module")
  feature_columns = [image_column, ...]
  estimator = tf.estimator.LinearClassifier(feature_columns, ...)
  height, width = hub.get_expected_image_size(image_column.module_spec)
  input_fn = ...  # Provides "embeddings" with shape [None, height, width, 3].
  estimator.train(input_fn, ...)
```

#### Args:

* <b>`key`</b>: A string or `_FeatureColumn` identifying the input image data.
* <b>`module_spec`</b>: A string handle or a `ModuleSpec` identifying the module.


#### Returns:

`_DenseColumn` that converts from pixel data.


#### Raises:

* <b>`ValueError`</b>: if module_spec is not suitable for use in this feature column.