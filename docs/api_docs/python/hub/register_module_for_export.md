<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="hub.register_module_for_export" />
</div>

# hub.register_module_for_export

``` python
hub.register_module_for_export(
    module,
    export_name
)
```

Register a Module to be exported under `export_name`.


This function registers `module` to be exported by `LatestModuleExporter`
under a subdirectory named `export_name`.

Note that `export_name` must be unique for each module exported from the
current graph. It only controls the export subdirectory name and it has
no scope effects such as the `name` parameter during Module instantiation.

#### Args:

* <b>`module`</b>: Module instance to be exported.
* <b>`export_name`</b>: subdirectory name to use when performing the export.


#### Raises:

* <b>`ValueError`</b>: if `export_name` is already taken in the current graph.