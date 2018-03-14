<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="hub.get_expected_image_size" />
</div>

# hub.get_expected_image_size

``` python
hub.get_expected_image_size(
    module_or_spec,
    signature=None,
    input_name=None
)
```

Returns expected [height, width] dimensions of an image input.

#### Args:

* <b>`module_or_spec`</b>: a Module or ModuleSpec that accepts image inputs.
* <b>`signature`</b>: a string with the key of the signature in question.
    If None, the default signature is used.
* <b>`input_name`</b>: a string with the input name for images. If None, the
    conventional input name `images` for the default signature is used.


#### Returns:

A list if integers `[height, width]`.


#### Raises:

* <b>`ValueError`</b>: If the size information is missing or malformed.