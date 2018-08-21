<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="hub.create_module_spec_from_saved_model" />
<meta itemprop="path" content="stable" />
</div>

# hub.create_module_spec_from_saved_model

``` python
hub.create_module_spec_from_saved_model(
    saved_model_path,
    drop_collections=None
)
```

Experimental: Create a ModuleSpec out of a SavedModel.

Define a ModuleSpec from a SavedModel. Note that this is not guaranteed to
work in all cases and it assumes the SavedModel has followed some conventions:

- The serialized SaverDef can be ignored and instead can be reconstructed.
- The init op and main op can be ignored and instead the module can be
  initialized by using the conventions followed by
  `tf.train.MonitoredSession`.

Note that the set of features supported can increase over time and have side
effects that were not previously visible. The pattern followed to avoid
surprises is forcing users to declare which features to ignore (even
if they are not supported).

Note that this function creates a ModuleSpec that when exported exports a
Module (based on a modified copy of the original SavedModel) and not a
SavedModel.

#### Args:

* <b>`saved_model_path`</b>: Directory with the SavedModel to use.
* <b>`drop_collections`</b>: Additionally list of collection to drop.


#### Returns:

A ModuleSpec.