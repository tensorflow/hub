<!--* freshness: { owner: 'kempy' } *-->

# TensorFlow Hub

TensorFlow Hub is an open repository and library for reusable machine learning.
The [tfhub.dev](https://tfhub.dev) repository provides many pre-trained models:
text embeddings, image classification models, TFJS/TFLite models and much more.
The repository is open to
[community contributors](https://tfhub.dev/s?subtype=publisher).

The [`tensorflow_hub`](https://github.com/tensorflow/hub) library lets you
download and reuse them in your TensorFlow program with a minimum amount of
code.

```python
import tensorflow_hub as hub

embed = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1")
embeddings = embed(["A long sentence.", "single-word", "http://example.com"])
print(embeddings.shape, embeddings.dtype)
```

## Next Steps

-   [Find models on tfhub.dev](https://tfhub.dev)
-   [Publish models on tfhub.dev](publish.md)
-   TensorFlow Hub library
    -   [Install TensorFlow Hub](installation.md)
    -   [Library overview](lib_overview.md)
