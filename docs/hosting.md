# Hosting a Module

TensorFlow Hub supports HTTP based distribution of modules. In particular
the protocol allows to use the URL identifying the module both as the
documentation of the module and the endpoint to fetch the module.


## Protocol

When a URL such as `https://example.com/module` is used to identify a
module to load or instantiate, the module resolver will attempt to
download a compressed tarball from the URL after appending a query
parameter `?tf-hub-format=compressed`.

The query param is to be interpreted as a comma separated list of the
module formats that the client is interested in. For now only the
"compressed" format is defined.

The **compressed** format indicates that the client expects a `tar.gz`
archive with the module contents. The root of the archive is the root
of the module directory and should contain a SavedModel, as in this
example:

```shell
# Create a compressed module from a SavedModel directory.
$ tar -cz -f module.tar.gz --owner=0 --group=0 -C /tmp/export-module/ .

# Inspect files inside a compressed module
$ tar -tf module.tar.gz
./
./variables/
./variables/variables.data-00000-of-00001
./variables/variables.index
./assets/
./saved_model.pb
```

Tarballs for use with the deprecated `hub.Module()` API from TF1 will also
contain a `./tfhub_module.pb` file. The `hub.load()` API for TF2 SavedModels
ignores such a file.

The `tensorflow_hub` library expects that module URLs are versioned
and that the module content of a given version is immutable, so that it can
be cached indefinitely.
