# Using a Module

## Instantiating a Module

A TensorFlow Hub module is imported into a TensorFlow program by
creating a `Module` object from a string with its URL or filesystem path,
such as:

```python
m = hub.Module("path/to/a/module_dir")
```

This adds the module's variables to the current TensorFlow graph.
Running their initializers will read their pre-trained values from disk.
Likewise, tables and other state is added to the graph.

## Caching Modules

When creating a module from a URL, the module content is downloaded
and cached in the local system temporary directory. The location where
modules are cached can be overridden using `TFHUB_CACHE_DIR` environment
variable.

For example, setting `TFHUB_CACHE_DIR` to `/my_module_cache`:

```shell
$ export TFHUB_CACHE_DIR=/my_module_cache
```

and then creating a module from a URL:

```python
m = hub.Module("https://tfhub.dev/google/progan-128/1")
```

results in downloading the unpacked version of the module in
`/my_module_cache`.


## Applying a Module

Once instantiated, a module `m` can be called zero or more times like a Python
function from tensor inputs to tensor outputs:

```python
y = m(x)
```

Each such call adds operations to the current TensorFlow graph to compute
`y` from `x`. If this involves variables with trained weights, these are
shared between all applications.

Modules can define multiple named *signatures* in order to allow being applied
in more than one way (similar to how Python objects have *methods*).
A module's documentation should describe the available
signatures. The call above applies the signature named `"default"`. Any
signature can be selected by passing its name to the optional `signature=`
argument.

If a signature has multiple inputs, they must be passed as a dict,
with the keys defined by the signature. Likewise, if a signature has
multiple outputs, these can be retrieved as a dict by passing `as_dict=True`,
under the keys defined by the signature. (The key `"default"` is for the
single output returned if `as_dict=False`.)
So the most general form of applying a Module looks like:

```python
outputs = m(dict(apples=x1, oranges=x2), signature="fruit_to_pet", as_dict=True)
y1 = outputs["cats"]
y2 = outputs["dogs"]
```

A caller must supply all inputs defined by a signature, but there is no
requirement to use all of a module's outputs.
TensorFlow will run only those parts of the module that end up
as dependencies of a target in `tf.Session.run()`. Indeed, module publishers may
choose to provide various outputs for advanced uses (like activations of
intermediate layers) along with the main outputs. Module consumers should
handle additional outputs gracefully.
