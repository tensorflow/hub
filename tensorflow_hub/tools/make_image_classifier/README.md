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

This tool requires tensorflow and tensorflow_hub libraries,
which can be installed with:

```shell
$ pip install "tensorflow~=2.0"
$ pip install "tensorflow-hub[make_image_classifier]~=0.6"
```

After installation, the `make_image_classifier` executable is available
on the command line:

```shell
$ make_image_classifier --help
```

This tool tends to run much faster with a GPU, if TensorFlow is installed
to use it. To do so, you need to install GPU drivers per
[tensorflow.org/install/gpu](https://www.tensorflow.org/install/gpu)
and use pip package `"tensorflow-gpu~=2.0"`.


## Basic Usage

Basic usage of the tool looks like

```shell
$ make_image_classifier \
  --image_dir my_image_dir \
  --tfhub_module https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4 \
  --image_size 224 \
  --saved_model_dir my_dir/new_model \
  --labels_output_file class_labels.txt \
  --tflite_output_file new_mobile_model.tflite \
  --summaries_dir my_log_dir
```

The `--image_dir` is a directory of subdirectories of images, defining
the classes you want your model to distinguish. Say you wanted to
classify your photos of pets to be cat, dog or rabbit. Then you would
arrange JPEG files of such photos in a directory structure like

```
my_image_dir
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
as daisy, dandelion, rose, sunflower or tulip.

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

After training, the given `--saved_model_dir` is created and filled
with several files that represent the complete image classification model
in TensorFlow's SavedModel format. This can be deployed to TensorFlow Serving.

If `--labels_output_file` is given, the names of the classes are written
to that text file, one per line, in the same order as they appear
in the predictions output by the model.

If `--tflite_output_file` is given, the complete image classification model
is written to that file in TensorFlow Lite's model format ("flatbuffers").
This can be deployed to TF Lite on mobile devices.
If you are not deploying to TF Lite, you can simply omit this flag.

If `--summaries_dir` is given, you can monitor your model training
on TensorBoard. See [this guide](https://www.tensorflow.org/tensorboard/get_started)
on how to enable TensorBoard.

If you set all the flags as in the example above, you can test the
resulting TF Lite model with
[tensorflow/lite/examples/python/label_image.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py)
by downloading that program and running on an image like

```shell
python label_image.py \
  --input_mean 0 --input_std 255 \
  --model_file new_mobile_model.tflite --label_file class_labels.txt \
  --image my_image_dir/cat/a_feline_photo.jpg  # <<< Adjust filename.
```


## Advanced usage

Additional command-line flags let you control the training process.
In particular, you can increase `--train_epochs` to train more,
and set the `--learning_rate` and `--momentum` for the SGD optimizer.

Also, you can set `--do_fine_tuning` to train the TensorFlow Hub
module together with the classifier.

There is other hyperparameters for regularization such as
`--l1_regularizer` and `--l2_regularizer`, and for data augmentations
such as `--rotation_range` and `--horizontal_flip`. Generally, the
default values can give a good performance. You can find a full list
of hyperparameters available in `make_image_classifier.py` and their
default values in `make_image_classifier_lib.py`.
