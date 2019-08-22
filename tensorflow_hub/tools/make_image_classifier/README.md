# Making your own TensorFlow model for image classification

The `make_image_classifier` tool comes with the tensorflow_hub library
and lets you build and train a TensorFlow model for image classification
from the command line, no coding required. The tool needs
a number of example images for each class (many dozens or hundreds),
but a default ("TF Flowers") is provided.

**Note:** This tool and its documentation are still under development.
It is meant to replace
[image_retraining/retrain.py](https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py).

If you are a developer looking for a coding example, please take a look at
[examples/colab/tf2_image_retraining.ipynb](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb)
which demonstrates the key techniques of this program in your browser.


## Installation

This tool requires TensorFlow 2.0 (or its public beta1) and
tensorflow_hub 0.6.0 or better (yet to be released as of August 2019).

```shell
$ pip install 'tensorflow>=2.0.0beta1'
$ pip install 'tensorflow-hub[make_image_classifier]>=0.6.0'
```

After that, the `make_image_classifier` executable is available on the
command line:

```shell
$ make_image_classifier --help
```

This tool tends to run much faster with a GPU, if TensorFlow is installed
to use it. For instructions, see
[tensorflow.org/install/gpu](https://www.tensorflow.org/install/gpu).


## Basic Usage

Basic usage of the tool looks like

```shell
$ make_image_classifier \
  --image_dir my_image dir \
  --tfhub_module https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4 \
  --image_size 224 \
  --saved_model_dir my_dir/new_model
```

The `--image_dir` is a directory of subdirectories of images, defining
the classes you want your model to distinguish. Say you wanted to
classify your photos of pets to be cat, dog or rabbit. Then you would
arrange JPEG files of such photos in a directory structure like

```
my image_dir
|-- cat
|   |-- a_feline_photo.jpg
|   |-- another_cat_pic.jpg
|   `-- ...
|-- dog
|   |-- PuppyInBasket.JPG
|   |-- walking_the_dog.jpeg
|   `-- ...
`-- rabbit
    |-- IMG87654321.JPG
    |-- my_fluffy_rabbit.JPEG
    `-- ...
```

Good training results need many images (many dozens, possibly hundreds
per class).

**Note:** For a quick demo, omit --image_dir. This will download and use
the "TF Flowers" dataset and train a model to classify photos of flowers
as dasiy, dandelion, rose, sunflower or tulip.

The `--tfhub_module` is the URL of a pre-trained model piece, or "module",
on [TensorFlow Hub](https://tfhub.dev). You can point your browser to the
module URL to see documentation for it. This tool requires a module
for image feature extraction in TF2 format. You can find them on TF Hub with
[this search](https://tfhub.dev/s?module-type=image-feature-vector&q=tf2).

Images are resized to the given `--image_size` after reading from
disk. It depends on the TF Hub module whether it accepts only a fixed size
(in which case you can omit this flag) or an arbitrary size (in which
case you should start off by setting this to the standard value
advertised in the module documentation).

Model training consumes your input data multiple times ("epochs").
Some part of the data is set aside as validation data; the partially
trained model is evaluated on that after each epoch. You can see
progress bars and accuracy indicators on the console.

After training, the given `--saved_model_dir`is created and filled
with several files that represent the complete image classification model
in TensorFlow's SavedModel format. This can be deployed to
TensorFlow Serving or TensorFlow Lite.


## Advanced usage

Additional command-line flags let you control the training process.
In particular, you can increase `--train_epochs` to train more,
and set the `--learning_rate` for the SGD optimizer.

Also, you can set `--do_fine_tuning` to train the TensorFlow Hub
module together with the classifier.
