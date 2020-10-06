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
TensorFlow Hub (in its TF2/SavedModel format, not the legacy TF1 Hub format)
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

import io
import tempfile

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow_hub.tools.make_image_classifier import make_image_classifier_lib as lib

_DEFAULT_HPARAMS = lib.get_default_hparams()

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
flags.DEFINE_string(
    "summaries_dir", None,
    "Where to save summary logs for TensorBoard.")
flags.DEFINE_float(
    "assert_accuracy_at_least", None,
    "If set, the program fails if the validation accuracy at the end of "
    "training is less than this number (between 0 and 1), and no export of "
    "the trained model happens.")
flags.DEFINE_integer(
    "train_epochs", _DEFAULT_HPARAMS.train_epochs,
    "Training will do this many iterations over the dataset.")
flags.DEFINE_bool(
    "do_fine_tuning", _DEFAULT_HPARAMS.do_fine_tuning,
    "If set, the --tfhub_module is trained together with the rest of "
    "the model being built.")
flags.DEFINE_integer(
    "batch_size", _DEFAULT_HPARAMS.batch_size,
    "Each training step samples a batch of this many images "
    "from the training data. (You may need to shrink this when using a GPU "
    "and getting out-of-memory errors. Avoid values below 8 when re-training "
    "modules that use batch normalization.)")
flags.DEFINE_float(
    "learning_rate", _DEFAULT_HPARAMS.learning_rate,
    "The learning rate to use for gradient descent training.")
flags.DEFINE_float(
    "momentum", _DEFAULT_HPARAMS.momentum,
    "The momentum parameter to use for gradient descent training.")
flags.DEFINE_float(
    "dropout_rate", _DEFAULT_HPARAMS.dropout_rate,
    "The fraction of the input units to drop, used in dropout layer.")
flags.DEFINE_bool(
    "set_memory_growth", False,
    "If flag is set, memory growth functionality flag will be set as true for "
    "all GPUs prior to training. "
    "More details: "
    "https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth"
)
flags.DEFINE_float(
    "l1_regularizer", _DEFAULT_HPARAMS.l1_regularizer,
    "Coefficient of L1 regularization applied on model weights.")
flags.DEFINE_float(
    "l2_regularizer", _DEFAULT_HPARAMS.l2_regularizer,
    "Coefficient of L2 regularization applied on model weights.")
flags.DEFINE_float("label_smoothing", _DEFAULT_HPARAMS.label_smoothing,
                   "Coefficient of label smoothing used in loss function.")
flags.DEFINE_float("validation_split", _DEFAULT_HPARAMS.validation_split,
                   "The fractin of the dataset splitted into a validation set")
flags.DEFINE_bool(
    "do_data_augmentation", _DEFAULT_HPARAMS.do_data_augmentation,
    "Whether do data augmentation on training set."
    "Can use default augmentation params or specifying them.")
flags.DEFINE_integer("rotation_range", _DEFAULT_HPARAMS.rotation_range,
                     "Degree range for random rotation.")
flags.DEFINE_bool("horizontal_flip", _DEFAULT_HPARAMS.horizontal_flip,
                  "Horizontally flip images.")
flags.DEFINE_float(
    "width_shift_range", _DEFAULT_HPARAMS.width_shift_range,
    "Shift images horizontally by pixels(if >=1) or by ratio(if <1).")
flags.DEFINE_float(
    "height_shift_range", _DEFAULT_HPARAMS.height_shift_range,
    "Shift images vertically by pixels(if >=1) or by ratio(if <1).")
flags.DEFINE_float("shear_range", _DEFAULT_HPARAMS.shear_range,
                   "Shear angle in counter-clockwise direction in degrees.")
flags.DEFINE_float("zoom_range", _DEFAULT_HPARAMS.zoom_range,
                   "Range for random zoom.")
flags.DEFINE_enum("distribution_strategy", None, ["", "mirrored"],
                  "The distribution strategy the classifier should use.")
FLAGS = flags.FLAGS


def _get_hparams_from_flags():
  """Creates dict of hyperparameters from flags."""
  return lib.HParams(
      train_epochs=FLAGS.train_epochs,
      do_fine_tuning=FLAGS.do_fine_tuning,
      batch_size=FLAGS.batch_size,
      learning_rate=FLAGS.learning_rate,
      momentum=FLAGS.momentum,
      dropout_rate=FLAGS.dropout_rate,
      l1_regularizer=FLAGS.l1_regularizer,
      l2_regularizer=FLAGS.l2_regularizer,
      label_smoothing=FLAGS.label_smoothing,
      validation_split=FLAGS.validation_split,
      do_data_augmentation=FLAGS.do_data_augmentation,
      rotation_range=FLAGS.rotation_range,
      horizontal_flip=FLAGS.horizontal_flip,
      width_shift_range=FLAGS.width_shift_range,
      height_shift_range=FLAGS.height_shift_range,
      shear_range=FLAGS.shear_range,
      zoom_range=FLAGS.zoom_range)


def _check_keras_dependencies():
  """Checks dependencies of tf.keras.preprocessing.image are present.

  This function may come to depend on flag values that determine the kind
  of preprocessing being done.

  Raises:
    ImportError: If dependencies are missing.
  """
  try:
    tf.keras.preprocessing.image.load_img(io.BytesIO())
  except ImportError:
    print("\n*** Unsatisfied dependencies of keras_preprocessing.image. ***\n"
          "To install them, use your system's equivalent of\n"
          "pip install tensorflow_hub[make_image_classifier]\n")
    raise
  except Exception as e:  # pylint: disable=broad-except
    # Loading from dummy content as above is expected to fail in other ways.
    pass


def _assert_accuracy(train_result, assert_accuracy_at_least):
  # Fun fact: With TF1 behavior, the key was called "val_acc".
  val_accuracy = train_result.history["val_accuracy"][-1]
  accuracy_message = "found {:f}, expected at least {:f}".format(
      val_accuracy, assert_accuracy_at_least)
  if val_accuracy >= assert_accuracy_at_least:
    print("ACCURACY PASSED:", accuracy_message)
  else:
    raise AssertionError("ACCURACY FAILED:", accuracy_message)


def _set_gpu_memory_growth():
  # Original code reference found here:
  # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
  gpus = tf.config.experimental.list_physical_devices("GPU")
  if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    print("All GPUs will scale memory steadily")
  else:
    print("No GPUs found for set_memory_growth")


def main(args):
  """Main function to be called by absl.app.run() after flag parsing."""
  del args
  _check_keras_dependencies()
  hparams = _get_hparams_from_flags()

  image_dir = FLAGS.image_dir or lib.get_default_image_dir()

  if FLAGS.set_memory_growth:
    _set_gpu_memory_growth()

  model, labels, train_result = lib.make_image_classifier(
      FLAGS.tfhub_module, image_dir, hparams,
      lib.get_distribution_strategy(FLAGS.distribution_strategy),
      FLAGS.image_size, FLAGS.summaries_dir)
  if FLAGS.assert_accuracy_at_least:
    _assert_accuracy(train_result, FLAGS.assert_accuracy_at_least)
  print("Done with training.")

  if FLAGS.labels_output_file:
    with tf.io.gfile.GFile(FLAGS.labels_output_file, "w") as f:
      f.write("\n".join(labels + ("",)))
    print("Labels written to", FLAGS.labels_output_file)

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
  logging.info("Running with tensorflow %s and hub %s",
               tf.__version__, hub.__version__)
  if not tf.executing_eagerly():
    raise ImportError("Sorry, this program needs TensorFlow 2.")


def run_main():
  """Entry point equivalent to executing this file."""
  _ensure_tf2()
  app.run(main)


if __name__ == "__main__":
  run_main()
