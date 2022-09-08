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

This library provides the major pieces for make_image_classifier (see there).
"""

import collections
import contextlib

import tensorflow as tf
import tensorflow_hub as hub

_DEFAULT_IMAGE_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"


def get_default_image_dir():
  """Returns the path to a default image dataset, downloading it if needed."""
  return tf.keras.utils.get_file("flower_photos",
                                 _DEFAULT_IMAGE_URL, untar=True)


class NoStrategy:
  scope = contextlib.contextmanager(lambda _: iter(range(1)))


def get_distribution_strategy(distribution_strategy_name):
  if distribution_strategy_name == "mirrored":
    return tf.distribute.MirroredStrategy()
  elif not distribution_strategy_name:
    return NoStrategy()
  else:
    raise ValueError(
        "Unknown distribution strategy {}".format(distribution_strategy_name))


class HParams(
    collections.namedtuple("HParams", [
        "train_epochs", "do_fine_tuning", "batch_size", "learning_rate",
        "momentum", "dropout_rate", "l1_regularizer", "l2_regularizer",
        "label_smoothing", "validation_split", "do_data_augmentation",
        "rotation_range", "horizontal_flip", "width_shift_range",
        "height_shift_range", "shear_range", "zoom_range"
    ])):
  """The hyperparameters for make_image_classifier.

  train_epochs: Training will do this many iterations over the dataset.
  do_fine_tuning: If true, the Hub module is trained together with the
    classification layer on top.
  batch_size: Each training step samples a batch of this many images.
  learning_rate: The learning rate to use for gradient descent training.
  momentum: The momentum parameter to use for gradient descent training.
  dropout_rate: The fraction of the input units to drop, used in dropout layer.
  """


def get_default_hparams():
  """Returns a fresh HParams object initialized to default values."""
  return HParams(
      train_epochs=5,
      do_fine_tuning=False,
      batch_size=32,
      learning_rate=0.005,
      momentum=0.9,
      dropout_rate=0.2,
      l1_regularizer=0.0,
      l2_regularizer=0.0001,
      label_smoothing=0.1,
      validation_split=0.2,
      do_data_augmentation=False,
      rotation_range=40,
      horizontal_flip=True,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0,
      zoom_range=0.2)


def _get_data_with_keras(image_dir, image_size, batch_size, validation_split,
                         do_data_augmentation, augmentation_params):
  """Gets training and validation data via keras_preprocessing.

  Args:
    image_dir: A Python string with the name of a directory that contains
      subdirectories of images, one per class.
    image_size: A list or tuple with 2 Python integers specifying
      the fixed height and width to which input images are resized.
    batch_size: A Python integer with the number of images per batch of
      training and validation data.
    validation_split: A float representing the fraction of the dataset split
      into a validation set.
    do_data_augmentation: An optional boolean, controlling whether the
      training dataset is augmented by randomly distorting input images.
    augmentation_params: A dictionary containing the augmentation params as keys
      and their respective values.

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
  datagen_kwargs = dict(rescale=1. / 255, validation_split=validation_split)
  dataflow_kwargs = dict(target_size=image_size, batch_size=batch_size,
                         interpolation="bilinear")

  valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      **datagen_kwargs)
  valid_generator = valid_datagen.flow_from_directory(
      image_dir, subset="validation", shuffle=False, **dataflow_kwargs)

  if do_data_augmentation and len(augmentation_params):
    datagen_kwargs.update(**augmentation_params)
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
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


def _get_data_as_datasets(image_dir, image_size, hparams):
  """Gets training and validation data via tf.data.Dataset.

  Args:
    image_dir: A Python string with the name of a directory that contains
      subdirectories of images, one per class.
    image_size: A list or tuple with 2 Python integers specifying the fixed
      height and width to which input images are resized.
    hparams: A HParams object with hyperparameters controlling the training.

  Returns:
    A nested tuple ((train_data, train_size),
                    (valid_data, valid_size), labels) where:
    train_data, valid_data: tf.data.Dataset for use with Model.fit, each
      yielding batch of tuples (images, labels) where
        images is a float32 Tensor of shape [batch_size, height, width, 3]
          with pixel values in range [0,1],
        labels is a float32 Tensor of shape [batch_size, num_classes]
          with one-hot encoded classes.
    train_size, valid_size: Python integers with the numbers of training
      and validation examples, respectively.
    labels: A tuple of strings with the class labels (subdirectory names).
      The index of a label in this tuple is the numeric class id.
  """
  # Check if hparam.shear_range is set. If yes, throw an error since shear is
  # not supported when using preprocessing layers.
  if hparams.shear_range != 0:
    raise ValueError("Found non-zero value for shear_range. Shear is not "
                     "supported when using reading input with tf.data.Dataset "
                     "and using preprocessing layers.")

  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      image_dir,
      validation_split=hparams.validation_split,
      subset="training",
      label_mode="categorical",
      # Seed needs to provided when using validation_split and shuffle = True.
      # A fixed seed is used so that the validation set is stable across runs.
      seed=123,
      image_size=image_size,
      batch_size=1)
  class_names = tuple(train_ds.class_names)
  train_size = train_ds.cardinality().numpy()
  train_ds = train_ds.unbatch().batch(hparams.batch_size)
  train_ds = train_ds.repeat()

  normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(
      1. / 255)
  preprocessing_model = tf.keras.Sequential([normalization_layer])
  if hparams.do_data_augmentation:
    preprocessing_model.add(
        tf.keras.layers.experimental.preprocessing.RandomRotation(
            hparams.rotation_range))
    preprocessing_model.add(
        tf.keras.layers.experimental.preprocessing.RandomTranslation(
            0, hparams.width_shift_range))
    preprocessing_model.add(
        tf.keras.layers.experimental.preprocessing.RandomTranslation(
            hparams.height_shift_range, 0))
    # Like the old tf.keras.preprocessing.image.ImageDataGenerator(),
    # image sizes are fixed when reading, and then a random zoom is applied.
    # If all training inputs are larger than image_size, one could also use
    # RandomCrop with a batch size of 1 and rebatch later.
    preprocessing_model.add(
        tf.keras.layers.experimental.preprocessing.RandomZoom(
            hparams.zoom_range, hparams.zoom_range))
    if hparams.horizontal_flip:
      preprocessing_model.add(
          tf.keras.layers.experimental.preprocessing.RandomFlip(
              mode="horizontal"))
  train_ds = train_ds.map(lambda images, labels:
                          (preprocessing_model(images), labels))

  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      image_dir,
      validation_split=hparams.validation_split,
      subset="validation",
      label_mode="categorical",
      seed=123,
      shuffle=False,
      image_size=image_size,
      batch_size=1)
  valid_size = val_ds.cardinality().numpy()
  val_ds = val_ds.unbatch().batch(hparams.batch_size)
  val_ds = val_ds.map(lambda images, labels:
                      (normalization_layer(images), labels))

  return ((train_ds, train_size), (val_ds, valid_size), class_names)


def _image_size_for_module(module_layer, requested_image_size=None):
  """Returns the input image size to use with the given module.

  Args:
    module_layer: A hub.KerasLayer initialized from a Hub module expecting
      image input.
    requested_image_size: An optional Python integer with the user-requested
      height and width of the input image; or None.

  Returns:
    A tuple (height, width) of Python integers that can be used as input
    image size for the given module_layer.

  Raises:
    ValueError: If requested_image_size is set but incompatible with the module.
    ValueError: If the module does not specify a particular input size and
       requested_image_size is not set.
  """
  # TODO(b/139530454): Use a library helper function once available.
  # The stop-gap code below assumes any concrete function backing the
  # module call will accept a batch of images with the one accepted size.
  module_image_size = tuple(
      module_layer._func.__call__  # pylint:disable=protected-access
      .concrete_functions[0].structured_input_signature[0][0].shape[1:3])
  if requested_image_size is None:
    if None in module_image_size:
      raise ValueError("Must specify an image size because "
                       "the selected TF Hub module specifies none.")
    else:
      return module_image_size
  else:
    requested_image_size = tf.TensorShape(
        [requested_image_size, requested_image_size])
    assert requested_image_size.is_fully_defined()
    if requested_image_size.is_compatible_with(module_image_size):
      return tuple(requested_image_size.as_list())
    else:
      raise ValueError("The selected TF Hub module expects image size {}, "
                       "but size {} is requested".format(
                           module_image_size,
                           tuple(requested_image_size.as_list())))


def build_model(module_layer, hparams, image_size, num_classes):
  """Builds the full classifier model from the given module_layer.

  If using a DistributionStrategy, call this under its `.scope()`.
  Args:
    module_layer: Pre-trained tfhub model layer.
    hparams: A namedtuple of hyperparameters. This function expects
      .dropout_rate: The fraction of the input units to drop, used in dropout
        layer.
    image_size: The input image size to use with the given module layer.
    num_classes: Number of the classes to be predicted.

  Returns:
    The full classifier model.
  """
  model = tf.keras.Sequential([
      tf.keras.Input(shape=(image_size[0], image_size[1], 3)), module_layer,
      tf.keras.layers.Dropout(rate=hparams.dropout_rate),
      tf.keras.layers.Dense(
          num_classes,
          activation="softmax",
          kernel_regularizer=tf.keras.regularizers.l1_l2(
              l1=hparams.l1_regularizer, l2=hparams.l2_regularizer))
  ])
  print(model.summary())
  return model


def train_model(model,
                hparams,
                train_data_and_size,
                valid_data_and_size,
                log_dir=None):
  """Trains model with the given data and hyperparameters.

  If using a DistributionStrategy, call this under its `.scope()`.
  Args:
    model: The tf.keras.Model from _build_model().
    hparams: A namedtuple of hyperparameters. This function expects
      .train_epochs: a Python integer with the number of passes over the
        training dataset;
      .learning_rate: a Python float forwarded to the optimizer;
      .momentum: a Python float forwarded to the optimizer;
      .batch_size: a Python integer, the number of examples returned by each
        call to the generators.
    train_data_and_size: A (data, size) tuple in which data is training data to
      be fed in tf.keras.Model.fit(), size is a Python integer with the
      numbers of training.
    valid_data_and_size: A (data, size) tuple in which data is validation data
      to be fed in tf.keras.Model.fit(), size is a Python integer with the
      numbers of validation.
    log_dir: A directory to write logs for TensorBoard into (defaults to None,
      no logs will then be written).

  Returns:
    The tf.keras.callbacks.History object returned by tf.keras.Model.fit().
  """
  train_data, train_size = train_data_and_size
  valid_data, valid_size = valid_data_and_size
  loss = tf.keras.losses.CategoricalCrossentropy(
      label_smoothing=hparams.label_smoothing)
  model.compile(
      optimizer=tf.keras.optimizers.SGD(
          learning_rate=hparams.learning_rate, momentum=hparams.momentum),
      loss=loss,
      metrics=["accuracy"])
  steps_per_epoch = train_size // hparams.batch_size
  validation_steps = valid_size // hparams.batch_size
  callbacks = []
  if log_dir != None:
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                    histogram_freq=1))
  return model.fit(
      train_data,
      epochs=hparams.train_epochs,
      steps_per_epoch=steps_per_epoch,
      validation_data=valid_data,
      validation_steps=validation_steps,
      callbacks=callbacks)


def make_image_classifier(tfhub_module,
                          image_dir,
                          hparams,
                          distribution_strategy=None,
                          requested_image_size=None,
                          log_dir=None,
                          use_tf_data_input=False):
  """Builds and trains a TensorFLow model for image classification.

  Args:
    tfhub_module: A Python string with the handle of the Hub module.
    image_dir: A Python string naming a directory with subdirectories of images,
      one per class.
    hparams: A HParams object with hyperparameters controlling the training.
    distribution_strategy: The DistributionStrategy make_image_classifier is
      running with.
    requested_image_size: A Python integer controlling the size of images to
      feed into the Hub module. If the module has a fixed input size, this must
      be omitted or set to that same value.
    log_dir: A directory to write logs for TensorBoard into (defaults to None,
      no logs will then be written).
    use_tf_data_input: Whether to read input with a tf.data.Dataset and use TF
      ops for preprocessing.
  """
  augmentation_params = dict(
      rotation_range=hparams.rotation_range,
      horizontal_flip=hparams.horizontal_flip,
      width_shift_range=hparams.width_shift_range,
      height_shift_range=hparams.height_shift_range,
      shear_range=hparams.shear_range,
      zoom_range=hparams.zoom_range)

  with distribution_strategy.scope():
    module_layer = hub.KerasLayer(
        tfhub_module, trainable=hparams.do_fine_tuning)
    image_size = _image_size_for_module(module_layer, requested_image_size)
    print("Using module {} with image size {}".format(tfhub_module, image_size))
    if use_tf_data_input:
      train_data_and_size, valid_data_and_size, labels = _get_data_as_datasets(
          image_dir, image_size, hparams)
    else:
      train_data_and_size, valid_data_and_size, labels = _get_data_with_keras(
          image_dir, image_size, hparams.batch_size, hparams.validation_split,
          hparams.do_data_augmentation, augmentation_params)
    print("Found", len(labels), "classes:", ", ".join(labels))
    model = build_model(module_layer, hparams, image_size, len(labels))
    train_result = train_model(model, hparams, train_data_and_size,
                                valid_data_and_size, log_dir)
  return model, labels, train_result
