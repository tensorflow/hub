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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.keras.layers.preprocessing import image_preprocessing as img_prep

import pathlib
import os
import functools
import numpy as np

_DEFAULT_IMAGE_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
_DEFAULT_STRATEGY = tf.distribute.OneDeviceStrategy(
    '/gpu:0' if tf.config.list_physical_devices('GPU')
    else '/cpu:0')

def get_default_image_dir():
  """Returns the path to a default image dataset, downloading it if needed."""
  return tf.keras.utils.get_file("flower_photos",
                                 _DEFAULT_IMAGE_URL, untar=True)


class HParams(
    collections.namedtuple("HParams", [
        "train_epochs", "do_fine_tuning", "batch_size", "learning_rate",
        "momentum", "dropout_rate", "l1_regularizer", "l2_regularizer", 
        "label_smoothing", "validation_split", "cache", "do_data_augmentation", 
        "rotation_range", "horizontal_flip", "width_shift_range", 
        "height_shift_range", "shear_range", "zoom_range", "strategy"
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
      cache=False,
      do_data_augmentation=False,
      rotation_range=40,
      horizontal_flip=True,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      strategy='default')


def _get_label(file_path, class_names):
  """Obtain the label of an image from its path.

  Args:
    file_path: A path string to an image example.
    class_names: A np.array containing all class names.

  Returns:
    a one-hot encoded label.
  """
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == class_names


def _decode_img(img, image_size):
  """Decode an image, convert to float, and resize it to the expected size."""
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  # Rescale is no longer needed, as this conversion will automatically
  # scale to [0, 1] when dtype is float.
  img = tf.image.convert_image_dtype(img, tf.float32)
  return tf.image.resize(img, image_size)


def _process_path(file_path, image_size, class_names):
  """Map an image path string to a (image, label) pair.

  args:
    file_path: A path string to an image example.
    image_size: A list specifying the size of an image ([H, W]).
    class_names: A np.array containing all class names.

  Returns:
    an image, label pair.
  """
  label = _get_label(file_path, class_names)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = _decode_img(img, image_size)
  return img, label


def rotate(inputs, lower=-10, upper=10):
  """Randomly rotate a batch of images by a degree within [lower, upper]."""
  inputs_shape = tf.shape(inputs)
  batch_size = inputs_shape[0]
  h_axis, w_axis = 1, 2
  img_hd = tf.cast(inputs_shape[h_axis], tf.float32)
  img_wd = tf.cast(inputs_shape[w_axis], tf.float32)
  min_angle = lower / 360.0 * 2 * np.pi
  max_angle = upper / 360.0 * 2 * np.pi
  angles = tf.random.uniform(
      shape=[batch_size], 
      minval=min_angle, 
      maxval=max_angle)
  return img_prep.transform(
      inputs,
      img_prep.get_rotation_matrix(angles, img_hd, img_wd),
      interpolation='bilinear')


def zoom(inputs, h_lower=-0.2, h_upper=0.2):
  """Randomly zoom in a batch of images by a scale within [h_lower, h_upper].
  The height and width are zoomed by the same scale.
  """
  inputs_shape = tf.shape(inputs)
  batch_size = inputs_shape[0]
  h_axis, w_axis = 1, 2
  img_hd = tf.cast(inputs_shape[h_axis], tf.float32)
  img_wd = tf.cast(inputs_shape[w_axis], tf.float32)
  height_zoom = tf.random.uniform(
      shape=[batch_size, 1], 
      minval=1. + h_lower, 
      maxval=1. + h_upper)
  width_zoom = height_zoom
  zooms = tf.cast(
      tf.concat([width_zoom, height_zoom], axis=1), 
      dtype=inputs.dtype)
  return img_prep.transform(
      inputs, img_prep.get_zoom_matrix(zooms, img_hd, img_wd),
      interpolation='bilinear')


def image_augmentation(img, labels, augment_params):
  """Image data augmentation utilities based on tf.image.ops.

  Functions here is heavily borrowed from 
  `tf.keras.layers.experimental.preprocessing`. This is a workaround since 
  currently `tf.keras.layers.experimental.preprocessing` layers don't support
  DistributeStrategy. We can move the augmentation steps to preprocessing
  layers after they can be used inside DistributeStrategy. See
  https://github.com/tensorflow/tensorflow/issues/39991 to monitor the progress.

  Args:
    img: A batch of images whose shape is [batch_size, height, width, channel].
    labels: A batch of one-hot encoded labels.
    augment_params: A dict containing augmentation-related parameters.

  Returns:
    img: The same batch of images that are augmented.
    labels: The same batch of labels.
  """
  if augment_params['horizontal_flip']:
    img = tf.image.random_flip_left_right(img)
  if augment_params['rotation_range'] and augment_params['rotation_range'] != 0:
    upper = augment_params['rotation_range']
    img = rotate(img, -upper, upper)
  if augment_params['zoom_range'] and augment_params['zoom_range'] != 0:
    upper = augment_params['zoom_range']
    img = zoom(img, -upper, upper)
  return img, labels


def _get_data_with_keras(image_dir, image_size, batch_size, 
                             validation_split, cache,
                             do_data_augmentation, augment_params):
  """Gets training and validation data via tf.data.Dataset.

  Args:
    image_dir: A Python string with the name of a directory that contains
      subdirectories of images, one per class.
    image_size: A list or tuple with 2 Python integers specifying
      the fixed height and width to which input images are resized.
    batch_size: A Python integer with the number of images per batch of
      training and validation data.
    validation_split: A float in [0, 1] specifying the percentage of validation
      set to be splitted from the dataset.
    cache: A bool value specifying whether or not caching datasets. Caching a
      dataset can speed up the data loading but will take memory. Set it to
      true only if the dataset can be fit into memory"
    do_data_augmentation: A bool value specifying whether doing augmentation.
    augment_params: A dict containing augmentation-related parameters.

  Returns:
    A nested tuple ((train_data, train_size),
                    (valid_data, valid_size), labels) where:
    train_data, valid_data: tf.data.Dataset for use with Model.fit,
      each yielding a batch of tuples (images, labels) where
        images is a float32 Tensor of shape [batch_size, height, width, 3]
          with pixel values in range [0,1],
        labels is a float32 Tensor of shape [batch_size, num_classes]
          with one-hot encoded classes.
    train_size, valid_size: Python integers with the numbers of training
      and validation examples, respectively.
    labels: A tuple of strings with the class labels (subdirectory names).
      The index of a label in this tuple is the numeric class id.
  """
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  image_dir = pathlib.Path(image_dir)
  class_names = np.array([item.name for item in image_dir.glob('*') 
                          if item.name != 'LICENSE.txt'])
  image_count = len(list(image_dir.glob('*/*')))
  # 8 * batch_size is a good choice for shuffle buffer size.
  buffer_size = batch_size * 8
  # shuffle here to avoid val dataset containing a single type of examples.
  list_ds = tf.data.Dataset.list_files(str(image_dir/'*/*'), shuffle=True)
  labeled_ds = list_ds.map(
      functools.partial(
          _process_path, image_size=image_size, class_names=class_names), 
      num_parallel_calls=AUTOTUNE)
  valid_size = int(image_count * validation_split)
  train_size = image_count - valid_size
  valid_ds = labeled_ds.take(valid_size)
  train_ds = labeled_ds.skip(valid_size)
  if cache:
    valid_ds = valid_ds.cache()
    train_ds = train_ds.cache()

  valid_ds = valid_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
  train_ds = train_ds.shuffle(buffer_size=buffer_size).repeat()\
      .batch(batch_size).prefetch(buffer_size=AUTOTUNE)

  if do_data_augmentation:
    train_ds = train_ds.map(
        functools.partial(
            image_augmentation, augment_params=augment_params),
        num_parallel_calls=AUTOTUNE)

  return (train_ds, train_size), (valid_ds, valid_size), class_names


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
          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=hparams.l1_regularizer,
                                                         l2=hparams.l2_regularizer))
  ])
  print(model.summary())
  return model


def train_model(model, hparams, train_data_and_size, valid_data_and_size,
                log_dir=None):
  """Trains model with the given data and hyperparameters.

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
  loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=hparams.label_smoothing)
  steps_per_epoch = train_size // hparams.batch_size
  validation_steps = valid_size // hparams.batch_size
  callbacks = []
  if log_dir != None:
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                    histogram_freq=1))
  model.compile(
      optimizer=tf.keras.optimizers.SGD(
          lr=hparams.learning_rate, momentum=hparams.momentum),
      loss=loss,
      metrics=["accuracy"])
  return model.fit(
      train_data,
      epochs=hparams.train_epochs,
      steps_per_epoch=steps_per_epoch,
      validation_data=valid_data,
      validation_steps=validation_steps,
      callbacks=callbacks)


def make_image_classifier(tfhub_module, image_dir, hparams,
                          requested_image_size=None,
                          log_dir=None):
  """Builds and trains a TensorFLow model for image classification.

  Args:
    tfhub_module: A Python string with the handle of the Hub module.
    image_dir: A Python string naming a directory with subdirectories of images,
      one per class.
    hparams: A HParams object with hyperparameters controlling the training.
    requested_image_size: A Python integer controlling the size of images to
      feed into the Hub module. If the module has a fixed input size, this
      must be omitted or set to that same value.
    log_dir: A directory to write logs for TensorBoard into (defaults to None,
      no logs will then be written).
  """
  if hparams.strategy == 'default':
    strategy = _DEFAULT_STRATEGY
  elif hparams.strategy == 'mirroredstrategy':
    strategy = tf.distribute.MirroredStrategy()
  else:
    raise NotImplementedError("Currently only support OneDeviceStrategy(default)"
      "and MirroredStrategy(mirroredstrategy).")

  # scale batch size according to replicas available
  if strategy.num_replicas_in_sync > 1:
    batch_size = hparams.batch_size * strategy.num_replicas_in_sync
    print("Found {} replicas, increase the batch size from {} to {}".format(
      strategy.num_replicas_in_sync, hparams.batch_size, batch_size
    ))
    hparams = hparams._replace(batch_size=batch_size)

  augment_params = dict(rotation_range=hparams.rotation_range,
    horizontal_flip=hparams.horizontal_flip,
    width_shift_range=hparams.width_shift_range,
    height_shift_range=hparams.height_shift_range,
    shear_range=hparams.shear_range,
    zoom_range=hparams.zoom_range)

  with strategy.scope():
    module_layer = hub.KerasLayer(tfhub_module, trainable=hparams.do_fine_tuning)
    image_size = _image_size_for_module(module_layer, requested_image_size)
    print("Using module {} with image size {}".format( tfhub_module, image_size))

    train_data_and_size, valid_data_and_size, labels = _get_data_with_keras(
        image_dir, image_size, hparams.batch_size, hparams.validation_split, 
        hparams.cache, hparams.do_data_augmentation, augment_params)
    print("Found", len(labels), "classes:", ", ".join(labels))
    print("Dataset size: %s (training) %s (validation)" % 
        (train_data_and_size[1], valid_data_and_size[1]))

    model = build_model(module_layer, hparams, image_size, len(labels))
    train_result = train_model(model, hparams, train_data_and_size,
                              valid_data_and_size, log_dir)
  return model, labels, train_result
