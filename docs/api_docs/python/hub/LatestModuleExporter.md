<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="hub.LatestModuleExporter" />
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="export"/>
</div>

# hub.LatestModuleExporter

## Class `LatestModuleExporter`



Regularly exports registered modules into timestamped directories.

Modules can be registered to be exported by this class by calling
`register_module_for_export` when constructing the graph. The
`export_name` provided determines the subdirectory name used when
exporting.

In addition to exporting, this class also garbage collects older exports.

Example use with EvalSpec:

```python
  train_spec = tf.estimator.TrainSpec(...)
  eval_spec = tf.estimator.EvalSpec(
      input_fn=eval_input_fn,
      exporters=[
          hub.LatestModuleExporter("tf_hub", serving_input_fn),
      ])
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
```

See `LatestModuleExporter.export()` for a direct use example.

## Properties

<h3 id="name"><code>name</code></h3>





## Methods

<h3 id="__init__"><code>__init__</code></h3>

``` python
__init__(
    name,
    serving_input_fn,
    exports_to_keep=5
)
```

Creates an `Exporter` to use with `tf.estimator.EvalSpec`.

#### Args:

* <b>`name`</b>: unique name of this `Exporter`, which will be used in the export
    path.
* <b>`serving_input_fn`</b>: A function with no arguments that returns a
    ServingInputReceiver. This is used with the `estimator` passed
    to `export()` to build the graph (in PREDICT mode) that registers the
    modules for export. The model in that graph is never run, so the actual
    data provided by this input fn does not matter.
* <b>`exports_to_keep`</b>: Number of exports to keep. Older exports will be garbage
    collected. Defaults to 5. Set to None to disable garbage collection.


#### Raises:

* <b>`ValueError`</b>: if any argument is invalid.

<h3 id="export"><code>export</code></h3>

``` python
export(
    estimator,
    export_path,
    checkpoint_path,
    eval_result=None,
    is_the_final_export=None
)
```

Actually performs the export of registered Modules.

This method creates a timestamped directory under `export_path`
with one sub-directory (named `export_name`) per module registered
via `register_module_for_export`.

Example use:

```python
  estimator = ... (Create estimator with modules registered for export)...
  exporter = hub.LatestModuleExporter("tf_hub", serving_input_fn)
  exporter.export(estimator, export_path, estimator.latest_checkpoint())
```

#### Args:

* <b>`estimator`</b>: the `Estimator` from which to export modules.
* <b>`export_path`</b>: A string containing a directory where to write the export
    timestamped directories.
* <b>`checkpoint_path`</b>: The checkpoint path to export.
* <b>`eval_result`</b>: Unused.
* <b>`is_the_final_export`</b>: Unused.


#### Returns:

The path to the created timestamped directory containing the exported
modules.



