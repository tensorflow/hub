<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="hub.resolve" />
<meta itemprop="path" content="Stable" />
</div>

# hub.resolve

``` python
hub.resolve(handle)
```

Resolves a module handle into a path.

 Resolves a module handle into a path by downloading and caching in
 location specified by TF_HUB_CACHE_DIR if needed.

#### Args:

* <b>`handle`</b>: (string) the Module handle to resolve.


#### Returns:

A string representing the Module path.