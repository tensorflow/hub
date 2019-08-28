# Copyright 2019 The TensorFlow Hub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Trains a TensorFlow model based on directories of images.

This program builds, trains and exports a TensorFlow 2.x model that classifies
natural images (photos) into a fixed set of classes. The classes are learned
from a user-supplied dataset of images, stored as a directory of subdirectories
of JPEG images, each subdirectory representing one class.

The model is built from a pre-trained image feature vector module from
TensorFlow Hub (in its TF2/SavedModel format, not the older hub.Module format)
followed by a linear classifier. The linear classifier, and optionally also
the TF Hub module, are trained on the new dataset. TF Hub offers a variety of
suitable modules with various size/accuracy tradeoffs.

The resulting model can be exported in TensorFlow's standard SavedModel format
and as a .tflite file for deployment to mobile devices with TensorFlow Lite.
TODO(b/139467904): Add support for post-training model optimization.

For more information, please see the README file next to the source code,
https://github.com/tensorflow/hub/blob/master/tensorflow_hub/tools/make_image_classifier/README.md
"""

# NOTE: This is an expanded, command-line version of
# https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb
# PLEASE KEEP THEM IN SYNC, such that running tests for this program
# provides assurance that the code in the colab notebook works.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from absl import app
from absl import flags
from absl import logging
import six
import tensorflow as tf
import tensorflow_hub as hub


flags.DEFINE_string(
    "image_dir", None,
    "A directory with subdirectories of images, one per class. "
    "If unset, the TensorFlow Flowers example dataset will be used. "
    "Internally, the dataset is split into training and validation pieces.")
flags.DEFINE_string(
    "tfhub_module",
    "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
    "Which TF Hub module to use. Must be a module in TF2/SavedModel format "
    "for computing image feature vectors.")
flags.DEFINE_integer(
    "image_size", None,
    "The height and width of images to feed into --tfhub_module. "
    "(For now, must be set manually for modules with variable input size.)")
flags.DEFINE_string(
    "saved_model_dir", None,
    "The final model is exported as a SavedModel directory with this name.")
flags.DEFINE_string(
    "tflite_output_file", None,
    "The final model is exported as a .tflite flatbuffers file with this name.")
flags.DEFINE_string(
    "labels_output_file", None,
    "Where to save the labels (that is, names of image subdirectories). "
    "The lines in this file appear in the same order as the predictions "
    "of the model.")
flags.DEFINE_integer(
    "train_epochs", 5,
    "Training will do this many iterations over the dataset.")
flags.DEFINE_bool(
    "do_fine_tuning", False,
    "If set, the --tfhub_module is trained together with the rest of "
    "the model being built.")
flags.DEFINE_integer(
    "batch_size", 32,
    "Each training step samples a batch of this many images "
    "from the training data. (You may need to shrink this when using a GPU "
    "and getting out-of-memory errors. Avoid values below 8 when re-training "
    "modules that use batch normalization.)")
flags.DEFINE_float(
    "learning_rate", 0.005,
    "The learning rate to use for gradient descent training.")
flags.DEFINE_float(
    "momentum", 0.9,
    "The momentum parameter to use for gradient descent training.")
flags.DEFINE_float(
    "assert_accuracy_at_least", None,
    "If set, the program fails if the validation accuracy at the end of "
    "training is less than this number (between 0 and 1), and no export of "
    "the trained model happens.")


FLAGS = flags.FLAGS

_DEFAULT_IMAGE_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"


def _check_keras_dependencies():
  """Checks dependencies of tf.keras.preprocessing.image are present.

  This function may come to depend on flag values that determine the kind
  of preprocessing being done.

  Raises:
    ImportError: If dependencies are missing.
  """
  try:
    tf.keras.preprocessing.image.load_img(six.BytesIO())
  except ImportError:
    print("\n*** Unsatisfied dependencies of keras_preprocessing.image. ***\n"
          "To install them, use your system's equivalent of\n"
          "pip install tensorflow_hub[make_image_classifier]\n")
    raise
  except Exception as e:  # pylint: disable=broad-except
    # Loading from dummy content as above is expected to fail in other ways.
    pass


def _get_data_with_keras(image_dir, image_size, batch_size,
                         do_data_augmentation=False):
  """Gets training and validation data via keras_preprocessing.

  Args:
    image_dir: A Python string with the name of a directory that contains
      subdirectories of images, one per class.
    image_size: A list or tuple with 2 Python integers specifying
      the fixed height and width to which input images are resized.
    batch_size: A Python integer with the number of images per batch of
      training and validation data.
    do_data_augmentation: An optional boolean, controlling whether the
      training dataset is augmented by randomly distorting input images.

  Returns:
    A nested tuple ((train_data, train_size),
                    (valid_data, valid_size), labels) where:
    train_data, valid_data: Generators for use with Model.fit_generator,
      each yielding tuples (images, labels) where
        images is a float32 Tensor of shape [batch_size, height, width, 3]
          with pixel values in range [0,1],
        labels is a float32 Tensor of shape [batch_size, num_classes]
          with one-hot encoded classes.
    train_size, valid_size: Python integers with the numbers of training
      and validation examples, respectively.
    labels: A tuple of strings with the class labels (subdirectory names).
      The index of a label in this tuple is the numeric class id.
  """
  datagen_kwargs = dict(rescale=1./255,
                        # TODO(b/139467904): Expose this as a flag.
                        validation_split=.20)
  dataflow_kwargs = dict(target_size=image_size, batch_size=batch_size,
                         interpolation="bilinear")

  valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      **datagen_kwargs)
  valid_generator = valid_datagen.flow_from_directory(
      image_dir, subset="validation", shuffle=False, **dataflow_kwargs)

  if do_data_augmentation:
    # TODO(b/139467904): Expose the following constants as flags.
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40, horizontal_flip=True, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
        **datagen_kwargs)
  else:
    train_datagen = valid_datagen
  train_generator = train_datagen.flow_from_directory(
      image_dir, subset="training", shuffle=True, **dataflow_kwargs)

  indexed_labels = [(index, label)
                    for label, index in train_generator.class_indices.items()]
  sorted_indices, sorted_labels = zip(*sorted(indexed_labels))
  assert sorted_indices == tuple(range(len(sorted_labels)))
  return ((train_generator, train_generator.samples),
          (valid_generator, valid_generator.samples),
          sorted_labels)


def _image_size_for_module(module_layer, flags_image_size=None):
  """Returns the input image size to use with the given module.

  Args:
    module_layer: A hub.KerasLayer initialized from a Hub module expecting
      image input.
    flags_image_size: An optional Python integer (supposedly from flags).
      If not None, requests using this value as height and width of input size.

  Returns:
    A tuple (height, width) of Python integers that can be used as input
    image size for the given module_layer.

  Raises:
    ValueError: If flags_image_size is set but incompatible with the module.
    ValueError: If the module does not specify a particular inpurt size and
       flags_image_size is not set.
  """
  # TODO(b/139530454): Use a library helper function once available.
  # The stop-gap code below assumes any concrete function backing the
  # module call will accept a batch of images with the one accepted size.
  module_image_size = tuple(
      module_layer._func.__call__  # pylint:disable=protected-access
      .concrete_functions[0].structured_input_signature[0][0].shape[1:3])
  if flags_image_size is None:
    if None in module_image_size:
      raise ValueError("Must set --image_size because "
                       "--tfhub_module specifies none.")
    else:
      return module_image_size
  else:
    requested_image_size = tf.TensorShape([flags_image_size, flags_image_size])
    assert requested_image_size.is_fully_defined()
    if requested_image_size.is_compatible_with(module_image_size):
      return tuple(requested_image_size.as_list())
    else:
      raise ValueError("--tfhub_module expects image size {}, "
                       "but --image_size requests {}".format(
                           module_image_size,
                           tuple(requested_image_size.as_list())))


def _build_model(module_layer, image_size, num_classes):
  """Builds the full classifier model from the given module_layer."""
  # TODO(b/139467904): Expose the hyperparameters below as flags.
  model = tf.keras.Sequential([
      module_layer,
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.Dense(num_classes, activation="softmax",
                            kernel_regularizer=tf.keras.regularizers.l2(0.0001))
  ])
  model.build((None, image_size[0], image_size[1], 3))
  print(model.summary())
  return model


def _train_model(model, train_epochs, learning_rate, momentum,
                 train_data_and_size, valid_data_and_size, batch_size):
  """Trains model with the given data and hyperparameters.

  Args:
    model: The tf.keras.Model from _build_model().
    train_epochs: A Python integer with the number of passes over the
      training dataset.
    learning_rate: A Python float forwarded to the optimizer.
    momentum: A Python float forwarded to the optimizer.
    train_data_and_size: A (data, size) tuple from _get_data_with_keras().
    valid_data_and_size: A (data, size) tuple from _get_data_with_keras().
    batch_size: A Python integer, the number of examples returned by each
      call to the generators.

  Returns:
    The tf.keras.callbacks.History object returned by tf.keras.Model.fit*().
  """
  train_data, train_size = train_data_and_size
  valid_data, valid_size = valid_data_and_size
  model.compile(
      optimizer=tf.keras.optimizers.SGD(lr=learning_rate, momentum=momentum),
      # TODO(b/139467904): Expose this hyperparameter as a flag.
      loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
      metrics=["accuracy"])
  steps_per_epoch = train_size // batch_size
  validation_steps = valid_size // batch_size
  return model.fit_generator(
      train_data, epochs=train_epochs, steps_per_epoch=steps_per_epoch,
      validation_data=valid_data, validation_steps=validation_steps)


def _assert_accuracy(train_result, assert_accuracy_at_least):
  # Fun fact: With TF1 behavior, the key was called "val_acc".
  val_accuracy = train_result.history["val_accuracy"][-1]
  accuracy_message = "found {:f}, expected at least {:f}".format(
      val_accuracy, assert_accuracy_at_least)
  if val_accuracy >= assert_accuracy_at_least:
    print("ACCURACY PASSED:", accuracy_message)
  else:
    raise AssertionError("ACCURACY FAILED:", accuracy_message)


def main(args):
  """Main function to be called by absl.app.run() after flag parsing."""
  del args
  _check_keras_dependencies()

  module_layer = hub.KerasLayer(FLAGS.tfhub_module,
                                trainable=FLAGS.do_fine_tuning)
  image_size = _image_size_for_module(module_layer, FLAGS.image_size)
  print("Using module {} with image size {}".format(
      FLAGS.tfhub_module, image_size))

  if FLAGS.image_dir:
    image_dir = FLAGS.image_dir
  else:
    print("No --image_dir given, downloading tf_flowers by default.")
    image_dir = tf.keras.utils.get_file(
        "flower_photos", _DEFAULT_IMAGE_URL, untar=True)
  train_data_and_size, valid_data_and_size, labels = _get_data_with_keras(
      image_dir, image_size, FLAGS.batch_size)
  print("Found", len(labels), "classes:", ", ".join(labels))

  if FLAGS.labels_output_file:
    with tf.io.gfile.GFile(FLAGS.labels_output_file, "w") as f:
      f.write("\n".join(labels + ("",)))
    print("Labels written to", FLAGS.labels_output_file)

  model = _build_model(module_layer, image_size, len(labels))
  train_result = _train_model(
      model, FLAGS.train_epochs, FLAGS.learning_rate, FLAGS.momentum,
      train_data_and_size, valid_data_and_size, FLAGS.batch_size)
  print("Done with training.")

  if FLAGS.assert_accuracy_at_least:
    _assert_accuracy(train_result, FLAGS.assert_accuracy_at_least)

  saved_model_dir = FLAGS.saved_model_dir
  if FLAGS.tflite_output_file and not saved_model_dir:
    # We need a SavedModel for conversion, even if the user did not request it.
    saved_model_dir = tempfile.mkdtemp()
  if saved_model_dir:
    tf.saved_model.save(model, saved_model_dir)
    print("SavedModel model exported to", saved_model_dir)

  if FLAGS.tflite_output_file:
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    lite_model_content = converter.convert()
    with tf.io.gfile.GFile(FLAGS.tflite_output_file, "wb") as f:
      f.write(lite_model_content)
    print("TFLite model exported to", FLAGS.tflite_output_file)


def _ensure_tf2():
  """Ensure running with TensorFlow 2 behavior.

  This function is safe to call even before flags have been parsed.

  Raises:
    ImportError: If tensorflow is too old for proper TF2 behavior.
  """
  logging.info("Running with tensorflow %s (git version %s) and hub %s",
               tf.__version__, tf.__git_version__, hub.__version__)
  if tf.__version__.startswith("1."):
    if tf.__git_version__ == "unknown":  # For internal testing use.
      try:
        tf.compat.v1.enable_v2_behavior()
        return
      except AttributeError:
        pass  # Fail below for missing enabler function.
    raise ImportError("Sorry, this program needs TensorFlow 2.")


def run_main():
  """Entry point equivalent to executing this file."""
  _ensure_tf2()
  app.run(main)


if __name__ == "__main__":
  run_main()
