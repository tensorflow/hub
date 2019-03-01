<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="hub" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__version__"/>
</div>

# Module: hub

TensorFlow Hub Library.

## Classes

[`class ImageModuleInfo`](./hub/ImageModuleInfo.md)

[`class KerasLayer`](./hub/KerasLayer.md): Wraps a Hub module (or a similar callable) for TF2 as a Keras Layer.

[`class LatestModuleExporter`](./hub/LatestModuleExporter.md): Regularly exports registered modules into timestamped directories.

[`class Module`](./hub/Module.md): Part of a TensorFlow model that can be transferred between models.

[`class ModuleSpec`](./hub/ModuleSpec.md): Represents the contents of a Module before it has been instantiated.

## Functions

[`add_signature(...)`](./hub/add_signature.md): Adds a signature to the module definition.

[`attach_image_module_info(...)`](./hub/attach_image_module_info.md): Attaches an ImageModuleInfo message from within a module_fn.

[`attach_message(...)`](./hub/attach_message.md): Adds an attached message to the module definition.

[`create_module_spec(...)`](./hub/create_module_spec.md): Creates a ModuleSpec from a function that builds the module's graph.

[`create_module_spec_from_saved_model(...)`](./hub/create_module_spec_from_saved_model.md): Experimental: Create a ModuleSpec out of a SavedModel.

[`get_expected_image_size(...)`](./hub/get_expected_image_size.md): Returns expected [height, width] dimensions of an image input.

[`get_num_image_channels(...)`](./hub/get_num_image_channels.md): Returns expected num_channels dimensions of an image input.

[`image_embedding_column(...)`](./hub/image_embedding_column.md): Uses a Module to get a dense 1-D representation from the pixels of images.

[`load(...)`](./hub/load.md): Loads a module from a handle.

[`load_module_spec(...)`](./hub/load_module_spec.md): Loads a ModuleSpec from the filesystem.

[`register_module_for_export(...)`](./hub/register_module_for_export.md): Register a Module to be exported under `export_name`.

[`resolve(...)`](./hub/resolve.md): Resolves a module handle into a path.

[`text_embedding_column(...)`](./hub/text_embedding_column.md): Uses a Module to construct a dense representation from a text feature.

## Other Members

<h3 id="__version__"><code>__version__</code></h3>

