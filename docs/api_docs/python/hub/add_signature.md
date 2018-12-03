<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="hub.add_signature" />
<meta itemprop="path" content="Stable" />
</div>

# hub.add_signature

``` python
hub.add_signature(
    name=None,
    inputs=None,
    outputs=None
)
```

Adds a signature to the module definition.

NOTE: This must be called within a `module_fn` that is defining a Module.

#### Args:

* <b>`name`</b>: Signature name as a string. If omitted, it is interpreted as 'default'
    and is the signature used when `Module.__call__` `signature` is not
    specified.
* <b>`inputs`</b>: A dict from input name to Tensor or SparseTensor to feed when
    applying the signature. If a single tensor is passed, it is interpreted
    as a dict with a single 'default' entry.
* <b>`outputs`</b>: A dict from output name to Tensor or SparseTensor to return from
    applying the signature. If a single tensor is passed, it is interpreted
    as a dict with a single 'default' entry.


#### Raises:

* <b>`ValueError`</b>: if the arguments are invalid.