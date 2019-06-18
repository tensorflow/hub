<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="hub.load" />
<meta itemprop="path" content="Stable" />
</div>

# hub.load

``` python
hub.load(
    handle,
    tags=None
)
```

Loads a module from a handle.

Currently this method is fully supported only with Tensorflow 2.x and with
modules created by calling tensorflow.saved_model.save(). The method works in
both eager and graph modes.

Depending on the type of handle used, the call may involve downloading a
Tensorflow Hub module to a local cache location specified by the
TFHUB_CACHE_DIR environment variable. If a copy of the module is already
present in the TFHUB_CACHE_DIR, the download step is skipped.

Currently, three types of module handles are supported:
  1) Smart URL resolvers such as tfhub.dev, e.g.:
     https://tfhub.dev/google/nnlm-en-dim128/1.
  2) A directory on a file system supported by Tensorflow containing module
     files. This may include a local directory (e.g. /usr/local/mymodule) or a
     Google Cloud Storage bucket (gs://mymodule).
  3) A URL pointing to a TGZ archive of a module, e.g.
     https://example.com/mymodule.tar.gz.

#### Args:

* <b>`handle`</b>: (string) the Module handle to resolve.
* <b>`tags`</b>: A set of strings specifying the graph variant to use, if loading from
    a v1 module.


#### Returns:

A trackable object (see tf.saved_model.load() documentation for details).


#### Raises:

* <b>`NotImplementedError`</b>: If the code is running against incompatible (1.x)
                       version of TF.