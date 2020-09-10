<!--* freshness: { owner: 'arnoegw' } reviewed: '2020-09-09' *-->

# Installation

## Installing tensorflow_hub

The `tensorflow_hub` library can be installed alongside TensorFlow 1 and
TensorFlow 2. We recommend that new users start with TensorFlow 2 right away,
and current users upgrade to it.

### Use with TensorFlow 2

Use [pip](https://pip.pypa.io/) to
[install TensorFlow 2](https://www.tensorflow.org/install) as usual. (See there
for extra instructions about GPU support.) Then install a current version of
[`tensorflow-hub`](https://pypi.org/project/tensorflow-hub/) next to it (must be
0.5.0 or newer).

```bash
$ pip install "tensorflow>=2.0.0"
$ pip install --upgrade tensorflow-hub
```

The TF1-style API of TensorFlow Hub works with the v1 compatibility mode of
TensorFlow 2.

### Legacy use with TensorFlow 1

The `tensorflow_hub` library requires TensorFlow version 1.7 or greater.

We strongly recommend to install it with TensorFlow 1.15, which defaults to
TF1-compatible behavior but contains many TF2 features under the hood to allow
some use of TensorFlow Hub's TF2-style APIs.

```bash
$ pip install "tensorflow>=1.15,<2.0"
$ pip install --upgrade tensorflow-hub
```

### Use of pre-release versions

The pip packages `tf-nightly` and `tf-hub-nightly` are built automatically from
the source code on github, with no release testing. This lets developers try out
the latest code without [building from source](build_from_source.md).

```bash
$ pip install tf-nightly
$ pip install --upgrade tf-hub-nightly
```

## Next Steps

-   [Library overview](lib_overview.md)
-   Tutorials:
    -   [Text classification](https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb)
    -   [Image classification](https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb)
    -   Additional examples
        [on GitHub](https://github.com/tensorflow/hub/blob/master/examples/README.md)
-   Find models on [tfhub.dev](https://tfhub.dev)
