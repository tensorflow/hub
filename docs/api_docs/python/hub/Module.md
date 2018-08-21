<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="hub.Module" />
<meta itemprop="path" content="stable" />
<meta itemprop="property" content="variable_map"/>
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="export"/>
<meta itemprop="property" content="get_attached_message"/>
<meta itemprop="property" content="get_input_info_dict"/>
<meta itemprop="property" content="get_output_info_dict"/>
<meta itemprop="property" content="get_signature_names"/>
</div>

# hub.Module

## Class `Module`



Part of a TensorFlow model that can be transferred between models.

A Module represents a part of a TensorFlow graph that can be exported to disk
(based on the SavedModel format) and later re-loaded. A Module has a defined
interface that allows it to be used in a replaceable way, with little or no
knowledge of its internals and its serialization format. Example:

```python
m = hub.Module("/tmp/text-embedding")
embeddings = m(sentences)
```

The module to instantiate is defined by its spec (a `ModuleSpec` or a
path where to load it from) which contains the module weights, assets and
signatures.

During instantiation the Module adds the state (e.g. variables and tables ops)
to the current graph. Afterwards, the method `__call__()` allows to apply the
module `signatures` multiple times, which adds ops for the computation.

A Module may provide different variants of its graph for different purposes
(say, training or serving, which may behave differently, e.g., for batch
normalization). Graph variants are identified by sets of string-valued tags.
The graph variant used to create a module that is exported must define all the
variables needed by any other graph variant that is subsequently used.

To make it possible to easily replace a module with another, they all assume
that they will be used with common TensorFlow conventions such as session
initialization and restore, use of collections for variables, regularization
losses and updates, etc.

## Properties

<h3 id="variable_map"><code>variable_map</code></h3>

Map from original variable names into tf.Variables (or lists of them).

This map translates between variable names relative to the module and the
corresponding Variable objects that have been created by instantiating it
in the current graph (with the applicable scoping added). Each key in the
map is a variable name as created by running the module's defining
`module_fn` in the root scope of an empty graph. Each value in the map is
a Variable object, or in case of partitioned variables a list of Variable
objects.

This property can be used with `tf.init_from_checkpoint` as `assignment_map`
in order to restore a pre-trained checkpoint into a Module before calling
`Module.export()`.

#### Returns:

A dict from the variable names in the Module to the instantiated
tf.Variables or list of tf.Variables (if partitioned). The keys of this
map are the same regardless of the scope of where the Module was
instantiated.



## Methods

<h3 id="__init__"><code>__init__</code></h3>

``` python
__init__(
    spec,
    trainable=False,
    name='module',
    tags=None
)
```

Constructs a Module to be used in the current graph.

This creates the module `state-graph` under an unused variable_scope based
on `name`. During this call a Module will:

- Add GLOBAL_VARIABLES under its scope. Those variables may be added to
  to the TRAINABLE_VARIABLES collection (depending on `trainable` parameter)
  and to the MODEL_VARIABLES. The variables must be initialized before use,
  and can be checkpointed as usual.

- Add ops to the INIT_TABLE_OPS collection, which must be run during session
  initialization and add constant tensors to ASSET_FILEPATHS that are
  needed during the execution of such ops.

- Add tensors to the REGULARIZATION_LOSSES collection (depending on
  `trainable` parameter).

#### Args:

* <b>`spec`</b>: A ModuleSpec defining the Module to instantiate or a path where
    to load a ModuleSpec from via `load_module_spec`.
* <b>`trainable`</b>: whether the Module is trainable. If False, no variables are
    added to TRAINABLE_VARIABLES collection, and no tensors are added to
    REGULARIZATION_LOSSES collection.
* <b>`name`</b>: A string, the variable scope name under which to create the Module.
    It will be uniquified and the equivalent name scope must be unused.
* <b>`tags`</b>: A set of strings specifying the graph variant to use.


#### Raises:

* <b>`RuntimeError`</b>: explaning the reason why it failed to instantiate the
    Module.
* <b>`ValueError`</b>: if the requested graph variant does not exists.

<h3 id="__call__"><code>__call__</code></h3>

``` python
__call__(
    inputs=None,
    _sentinel=None,
    signature=None,
    as_dict=None
)
```

Instantiates a module signature in the graph.

Example calls:

```python
  # Use default signature with one input and default output.
  embeddings = m(["hello world", "good morning"])

  # Use "encode" signature with one input and default output.
  encodings = m(["hello world"], signature="encode")

  # Use default signature with input dict and output dict.
  dict_outputs = m({"text": [...], "lang": [...]}, as_dict=True)
```

The method __call__() allows to create the graph ops that compute a
signature outputs given the inputs and using this module instance state.
Each signature can be applied multiple times with different inputs and they
all share the same module state.

A Module may define multiple signatures. Use `signature=<name>` to identify
the specific signature to instantiate. If omitted or None, the default
signature is used.

A signature may define various outputs. Use `as_dict=True` to return a dict
of all outputs. If omitted or False, the output named 'default' is
returned.

During this call a Module will:

- Add ops in the current name scope to convert the inputs in tensors to feed
  to the signature.

- Add ops to the UPDATE_OPS collection which depend on at least one of the
  provided inputs if the Module was constructed with `trainable=True`.

- Add constant tensors to ASSET_FILEPATHS, even if those are not needed
  directly needed for the signature.

#### Args:

* <b>`inputs`</b>: Inputs to the signature. A dict from input names to tensor
    values. If the signature only expects one input, one may pass
    a single value. If the signature has no inputs, it may be omitted.
* <b>`_sentinel`</b>: Used to prevent positional parameters besides `inputs`.
* <b>`signature`</b>: A string with the signature name to apply. If none, the
    default signature is used.
* <b>`as_dict`</b>: A boolean indicating whether to the return all the outputs
    of the signature as a dict or return only the default output.


#### Returns:

A tensor (single or sparse) if the signature defines a default output or
a dict from strings (output names) to tensors if `as_dict=True` is used.


#### Raises:

* <b>`TypeError`</b>: If there is a mismatch on arguments, inputs or outputs of
    the module signature.
* <b>`RuntimeError`</b>: If there are errors during creation of the signature graph.

<h3 id="export"><code>export</code></h3>

``` python
export(
    path,
    session
)
```

Exports the module with the variables from the session in `path`.

Note that it is the module definition in the ModuleSpec used to create this
module that gets exported. The session is only used to provide the value
of variables.

#### Args:

* <b>`path`</b>: path where to export the module to.
* <b>`session`</b>: session where to export the variables from.


#### Raises:

* <b>`RuntimeError`</b>: if there is an issue during the export.

<h3 id="get_attached_message"><code>get_attached_message</code></h3>

``` python
get_attached_message(
    key,
    message_type,
    required=False
)
```

Calls ModuleSpec.get_attached_message(); see there for more.

<h3 id="get_input_info_dict"><code>get_input_info_dict</code></h3>

``` python
get_input_info_dict(signature=None)
```

Describes the inputs required by a signature.

#### Args:

* <b>`signature`</b>: A string with the signature to get inputs information for.
    If None, the default signature is used if defined.


#### Returns:

The result of ModuleSpec.get_input_info_dict() for the given signature,
and the graph variant selected by `tags` when this Module was initialized.


#### Raises:

* <b>`KeyError`</b>: if there is no such signature.

<h3 id="get_output_info_dict"><code>get_output_info_dict</code></h3>

``` python
get_output_info_dict(signature=None)
```

Describes the outputs provided by a signature.

#### Args:

* <b>`signature`</b>: A string with the signature to get ouputs information for.
    If None, the default signature is used if defined.


#### Returns:

The result of ModuleSpec.get_input_info_dict() for the given signature,
and the graph variant selected by `tags` when this Module was initialized.


#### Raises:

* <b>`KeyError`</b>: if there is no such signature.

<h3 id="get_signature_names"><code>get_signature_names</code></h3>

``` python
get_signature_names()
```

Returns the module's signature names as an iterable of strings.



