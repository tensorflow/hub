# Caching model downloads from TF Hub

## Summary

The `tensorflow_hub` library caches models on the filesystem when they have been
downloaded from tfhub.dev (or other [hosting sites](hosting.md)) and
decompressed. The download location defaults to a local temporary directory but
can be customized by setting the environment variable `TFHUB_CACHE_DIR`
(recommended) or passing the command-line flag `--tfhub_cache_dir`. When using a
persistent location, be aware that there is no automatic cleanup.

The calls to `tensorflow_hub` functions in the actual Python code can and should
continue to use the canonical tfhub.dev URLs of models, which are portable
across systems and navigable for documentation.

## Specific execution environments

If and how the default `TFHUB_CACHE_DIR` needs changing depends on the execution
environment.

### Running locally on a workstation

For users running TensorFlow programs on their workstation, it should just work
in most cases to keep using the default location `/tmp/tfhub_modules`, or
whatever it is that Python returns for `os.path.join(tempfile.gettempdir(),
"tfhub_modules")`.

Users who prefer persistent caching across system reboots can instead set
`TFHUB_CACHE_DIR` to a location in their home directory. For example, a user of
the bash shell on a Linux system can add a line like the following to
`~/.bashrc`

```bash
export TFHUB_CACHE_DIR=$HOME/.cache/tfhub_modules
```

...restart the shell, and then this location will be used.

### Running on TPU in Colab notebooks

For running TensorFlow on CPU and GPU from within a
[Colab](https://colab.research.google.com/) notebook,
using the default local cache location should just work.

Running on TPU delegates to another machine that does not have access
to the default local cache location. Users with their own Google Cloud Storage
(GCS) bucket can work around this by setting a directory in that bucket as the
cache location with code like

```python
import os
os.environ["TFHUB_CACHE_DIR"] = "gs://my-bucket/tfhub-modules-cache"
```

...before calling the `tensorflow_hub` library.
