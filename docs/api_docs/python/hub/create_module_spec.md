<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="hub.create_module_spec" />
<meta itemprop="path" content="stable" />
</div>

# hub.create_module_spec

``` python
hub.create_module_spec(
    module_fn,
    tags_and_args=None,
    drop_collections=None
)
```

Creates a ModuleSpec from a function that builds the module's graph.

The `module_fn` is called on a new graph (not the current one) to build the
graph of the module and define its signatures via `hub.add_signature()`.
Example:

```python
# Define a text embedding module.
def my_text_module_fn():
  text_input = tf.placeholder(dtype=tf.string, shape=[None])
  embeddings = compute_embedding(text)
  hub.add_signature(inputs=text_input, outputs=embeddings)
```

See `add_signature()` for documentation on adding multiple input/output
signatures.

NOTE: In anticipation of future TF-versions, `module_fn` is called on a graph
that uses resource variables by default. If you want old-style variables then
you can use `with tf.variable_scope("", use_resource=False)` in `module_fn`.

Multiple graph variants can be defined by using the `tags_and_args` argument.
For example, the code:

```python
hub.create_module_spec(
    module_fn,
    tags_and_args=[({"train"}, {"is_training":True}),
                   (set(), {"is_training":False})])
```

calls `module_fn` twice, once as `module_fn(is_training=True)` and once as
`module_fn(is_training=False)` to define the respective graph variants:
for training with tags {"train"} and for inference with the empty set of tags.
Using the empty set aligns the inference case with the default in
Module.__init__().

#### Args:

* <b>`module_fn`</b>: a function to build a graph for the Module.
* <b>`tags_and_args`</b>: Optional list of tuples (tags, kwargs) of tags and keyword
    args used to define graph variants. If omitted, it is interpreted as
    [set(), {}], meaning `module_fn` is called once with no args.
* <b>`drop_collections`</b>: list of collection to drop.


#### Returns:

A ModuleSpec.


#### Raises:

* <b>`ValueError`</b>: if it fails to construct the ModuleSpec due to bad or
    unsupported values in the arguments or in the graphs constructed by
    `module_fn`.