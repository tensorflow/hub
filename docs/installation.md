# Installation and usage notes

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

## API stability

Although we hope to prevent breaking changes, this project is still under active
development and is not yet guaranteed to have a stable API or model format.

## Fairness

As in all of machine learning, [fairness](http://ml-fairness.com) is an
[important](https://research.googleblog.com/2016/10/equality-of-opportunity-in-machine.html)
consideration. Many pre-trained models are trained on large datasets. When
reusing any model, it’s important to be mindful of what data the model was
trained on (and whether there are any existing biases there), and how these
might impact your use of it.

## Security

Since they contain arbitrary TensorFlow graphs, models can be thought of as
programs.
[Using TensorFlow Securely](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md)
describes the security implications of referencing a model from an untrusted
source.
