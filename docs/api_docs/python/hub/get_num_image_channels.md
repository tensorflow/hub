<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="hub.get_num_image_channels" />
<meta itemprop="path" content="stable" />
</div>

# hub.get_num_image_channels

``` python
hub.get_num_image_channels(
    module_or_spec,
    signature=None,
    input_name=None
)
```

Returns expected num_channels dimensions of an image input.

This is for advanced users only who expect to handle modules with
image inputs that might not have the 3 usual RGB channels.

#### Args:

* <b>`module_or_spec`</b>: a Module or ModuleSpec that accepts image inputs.
* <b>`signature`</b>: a string with the key of the signature in question.
    If None, the default signature is used.
* <b>`input_name`</b>: a string with the input name for images. If None, the
    conventional input name `images` for the default signature is used.


#### Returns:

An integer with the number of input channels to the module.


#### Raises:

* <b>`ValueError`</b>: If the channel information is missing or malformed.