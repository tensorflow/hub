# Hosting your own models

TensorFlow Hub provides an open repository of trained models at
[thub.dev](https://tfhub.dev). The `tensorflow_hub` library can load models from
this repository and other HTTP based repositories of machine learning models. In
particular the protocol allows to use the URL identifying the model both for the
documentation of the model and the endpoint to fetch the model.

If you are interested in hosting your own repository of models that are loadable
with the `tensorflow_hub` library, your HTTP distribution service should follow
the following protocol.

## Protocol

When a URL such as `https://example.com/model` is used to identify a model to
load or instantiate, the model resolver will attempt to download a compressed
tarball from the URL after appending a query parameter
`?tf-hub-format=compressed`.

The query param is to be interpreted as a comma separated list of the model
formats that the client is interested in. For now only the "compressed" format
is defined.

The **compressed** format indicates that the client expects a `tar.gz` archive
with the model contents. The root of the archive is the root of the model
directory and should contain a SavedModel, as in this example:

```shell
# Create a compressed model from a SavedModel directory.
$ tar -cz -f model.tar.gz --owner=0 --group=0 -C /tmp/export-model/ .

# Inspect files inside a compressed model
$ tar -tf model.tar.gz
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

The `tensorflow_hub` library expects that model URLs are versioned and that the
model content of a given version is immutable, so that it can be cached
indefinitely.
