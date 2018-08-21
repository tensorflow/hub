<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="hub.attach_message" />
<meta itemprop="path" content="stable" />
</div>

# hub.attach_message

``` python
hub.attach_message(
    key,
    message
)
```

Adds an attached message to the module definition.

NOTE: This must be called within a `module_fn` that is defining a Module.

See ModuleSpec.get_attached_message() for an introduction to attached messages
and the API for module consumers.

To define a new type of attached message:

  * Select a reasonably descriptive name as a unique key. For now, keys must
    be valid Python identifiers that start with a letter. Punctuation besides
    underscores ('_') is reserved for future use in hierarchical names.

  * Define a Protocol Buffer message type to store the value for the key.
    (Use generic containers like google.protobuf.Value only if running
    the protocol compiler is infeasible for your build process.)

  * For module consumers, consider providing a small library that encapsulates
    the specific call to get_attached_message() behind a higher-level
    interface and supplies the right message type for parsing.

Attached messages work best for few messages of moderate size.
Avoid a large number of messages; use repetition within messages instead.
Avoid large messages (megabytes); consider module assets instead.

For modules with multiple graph versions, each graph version stores separately
what was attached from within the call to `module_fn` that defines its graph.

#### Args:

* <b>`key`</b>: A string with the unique key to retrieve this message. Must start
    with a letter and contain only letters, digits and underscores. If used
    repeatedly within one invocation of `module_fn`, then only the message
    from the final call will be returned by `get_attached_message()`.
* <b>`message`</b>: A protocol message object, to be stored in serialized form.


#### Raises:

* <b>`ValueError`</b>: if `key` is not a string of the form of a Python identifier.