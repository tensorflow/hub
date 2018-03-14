<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="hub" />
</div>

# Module: hub

TensorFlow Hub Library.

## Classes

[`class LatestModuleExporter`](./hub/LatestModuleExporter.md): Regularly exports registered modules into timestamped directories.

[`class Module`](./hub/Module.md): Part of a TensorFlow model that can be transferred between models.

[`class ModuleSpec`](./hub/ModuleSpec.md): Represents the contents of a Module before it has been instantiated.

## Functions

[`add_signature(...)`](./hub/add_signature.md): Adds a signature to the module definition.

[`create_module_spec(...)`](./hub/create_module_spec.md): Creates a ModuleSpec from a function that builds the module's graph.

[`get_expected_image_size(...)`](./hub/get_expected_image_size.md): Returns expected [height, width] dimensions of an image input.

[`get_num_image_channels(...)`](./hub/get_num_image_channels.md): Returns expected num_channels dimensions of an image input.

[`image_embedding_column(...)`](./hub/image_embedding_column.md): Uses a Module to get a dense 1-D representation from the pixels of images.

[`load_module_spec(...)`](./hub/load_module_spec.md): Loads a ModuleSpec from the filesystem.

[`register_module_for_export(...)`](./hub/register_module_for_export.md): Register a Module to be exported under `export_name`.

[`text_embedding_column(...)`](./hub/text_embedding_column.md): Uses a Module to construct a dense representation from a text feature.

