<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="hub.ModuleSpec" />
<meta itemprop="path" content="stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_attached_message"/>
<meta itemprop="property" content="get_input_info_dict"/>
<meta itemprop="property" content="get_output_info_dict"/>
<meta itemprop="property" content="get_signature_names"/>
<meta itemprop="property" content="get_tags"/>
</div>

# hub.ModuleSpec

## Class `ModuleSpec`



Represents the contents of a Module before it has been instantiated.

A ModuleSpec is the blueprint used by `Module` to create one or more instances
of a specific module in one or more graphs. The details on how to construct
the Module are internal to the library implementation but methods to inspect
a Module interface are public.

Note: Do not instantiate this class directly. Use <a href="../hub/load_module_spec.md"><code>hub.load_module_spec</code></a> or
<a href="../hub/create_module_spec.md"><code>hub.create_module_spec</code></a>.

## Methods

<h3 id="__init__"><code>__init__</code></h3>

``` python
__init__()
```

Do not instantiate directly.

<h3 id="get_attached_message"><code>get_attached_message</code></h3>

``` python
get_attached_message(
    key,
    message_type,
    tags=None,
    required=False
)
```

Returns the message attached to the module under the given key, or None.

Module publishers can attach protocol messages to modules at creation time
to provide module consumers with additional information, e.g., on module
usage or provenance (see see hub.attach_message()). A typical use would be
to store a small set of named values with modules of a certain type so
that a support library for consumers of such modules can be parametric
in those values.

This method can also be called on a Module instantiated from a ModuleSpec,
then `tags` are set to those used in module instatiation.

#### Args:

* <b>`key`</b>: A string with the key of an attached message.
* <b>`message_type`</b>: A concrete protocol message class (*not* object) used
    to parse the attached message from its serialized representation.
    The message type for a particular key must be advertised with the key.
* <b>`tags`</b>: Optional set of strings, specifying the graph variant from which
    to read the attached message.
* <b>`required`</b>: An optional boolean. Setting it true changes the effect of
    an unknown `key` from returning None to raising a KeyError with text
    about attached messages.


#### Returns:

An instance of `message_type` with the message contents attached to the
module, or `None` if `key` is unknown and `required` is False.


#### Raises:

* <b>`KeyError`</b>: if `key` is unknown and `required` is True.

<h3 id="get_input_info_dict"><code>get_input_info_dict</code></h3>

``` python
get_input_info_dict(
    signature=None,
    tags=None
)
```

Describes the inputs required by a signature.

#### Args:

* <b>`signature`</b>: A string with the signature to get inputs information for.
    If None, the default signature is used if defined.
* <b>`tags`</b>: Optional set of strings, specifying the graph variant to query.


#### Returns:

A dict from input names to objects that provide (1) a property `dtype`,
(2) a method `get_shape()` and (3) a read-only boolean property
`is_sparse`. The first two are compatible with the common API of Tensor
and SparseTensor objects.


#### Raises:

* <b>`KeyError`</b>: if there is no such signature or graph variant.

<h3 id="get_output_info_dict"><code>get_output_info_dict</code></h3>

``` python
get_output_info_dict(
    signature=None,
    tags=None
)
```

Describes the outputs provided by a signature.

#### Args:

* <b>`signature`</b>: A string with the signature to get ouputs information for.
    If None, the default signature is used if defined.
* <b>`tags`</b>: Optional set of strings, specifying the graph variant to query.


#### Returns:

A dict from input names to objects that provide (1) a property `dtype`,
(2) a method `get_shape()` and (3) a read-only boolean property
`is_sparse`. The first two are compatible with the common API of Tensor
and SparseTensor objects.


#### Raises:

* <b>`KeyError`</b>: if there is no such signature or graph variant.

<h3 id="get_signature_names"><code>get_signature_names</code></h3>

``` python
get_signature_names(tags=None)
```

Returns the module's signature names as an iterable of strings.

<h3 id="get_tags"><code>get_tags</code></h3>

``` python
get_tags()
```

Lists the graph variants as an iterable of set of tags.



