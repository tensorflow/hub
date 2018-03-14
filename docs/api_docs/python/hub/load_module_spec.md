<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="hub.load_module_spec" />
</div>

# hub.load_module_spec

``` python
hub.load_module_spec(path)
```

Loads a ModuleSpec from the filesystem.

#### Args:

* <b>`path`</b>: string describing the location of a module. There are several
        supported path encoding schemes:
        a) URL location specifying an archived module
          (e.g. http://domain/module.tgz)
        b) Any filesystem location of a module directory (e.g. /module_dir
           for a local filesystem). All filesystems implementations provided
           by Tensorflow are supported.


#### Returns:

A ModuleSpec.


#### Raises:

* <b>`ValueError`</b>: on unexpected values in the module spec.
* <b>`tf.OpError`</b>: on file handling exceptions.