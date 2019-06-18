<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="hub.sparse_text_embedding_column" />
<meta itemprop="path" content="Stable" />
</div>

# hub.sparse_text_embedding_column

``` python
hub.sparse_text_embedding_column(
    key,
    module_spec,
    combiner,
    default_value,
    trainable=False
)
```

Uses a Module to construct dense representations from sparse text features.

The input to this feature column is a batch of multiple strings with
arbitrary size, assuming the input is a SparseTensor.

This type of feature column is typically suited for modules that operate on
pre-tokenized text to produce token level embeddings which are combined with
the combiner into a text embedding. The combiner always treats the tokens as a
bag of words rather than a sequence.

The output (i.e., transformed input layer) is a DenseTensor, with shape
[batch_size, num_embedding_dim].

For Example:

```python
  comment = sparse_text_embedding_column("comment", "/tmp/text_module")
  feature_columns = [comment, ...]
  ...
  features = {
    "comment": tf.SparseTensor(indices=[[0, 0], [1, 2]],
                               values=['sparse', 'embedding'],
                               dense_shape=[3, 4]),
    ...
  }
  estimator = tf.estimator.DNNClassifier(hidden_units, feature_columns)
```

#### Args:

* <b>`key`</b>: A string or `_FeatureColumn` identifying the text feature.
* <b>`module_spec`</b>: A string handle or a `_ModuleSpec` identifying the module.
* <b>`combiner`</b>: a string specifying reducing op for embeddings in the same
    Example. Currently, 'mean', 'sqrtn', 'sum' are supported. Using
    combiner=None is undefined.
* <b>`default_value`</b>: default value for Examples where the text feature is empty.
    Note, it's recommended to have default_value consistent OOV tokens, in
    case there was special handling of OOV in the text module. If None, the
    text feature is assumed be non-empty for each Example.
* <b>`trainable`</b>: Whether or not the Module is trainable. False by default, meaning
    the pre-trained weights are frozen. This is different from the ordinary
    tf.feature_column.embedding_column(), but that one is intended for
    training from scratch.


#### Returns:

`_DenseColumn` that converts from text input.


#### Raises:

* <b>`ValueError`</b>: if module_spec is not suitable for use in this feature column.
* <b>`ValueError`</b>: if combiner not in ('mean', 'sqrtn', 'sum').