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
r"""Example for exporting a MNIST classifier in SavedModel v2.0 format.

The module has as a single signature, accepting a batch of images with shape
[None, 28, 28, 1] and returning a prediction vector.
In this example, we are loading the MNIST Dataset from TFDS and training a
simple digit classifier.

For a more realistic exporting example, see:
tensorflow/examples/saved_model/integration_tests/export_mnist_cnn.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import sys

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

FLAGS = None


class MNIST(tf.keras.models.Model):
  """Model representing a MNIST classifier."""

  def __init__(self, output_activation="softmax"):
    super(MNIST, self).__init__()
    self.layer_1 = tf.keras.layers.Dense(64)
    self.layer_2 = tf.keras.layers.Dense(10, activation=output_activation)

  def call(self, inputs):
    casted = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(inputs)
    flatten = tf.keras.layers.Flatten()(casted)

    def normalize_fn(x):
      return x / tf.reduce_max(tf.gather(x, 0))

    normalize = tf.keras.layers.Lambda(normalize_fn)(flatten)
    x = self.layer_1(normalize)
    output = self.layer_2(x)
    return output


def train_step(model, loss_fn, optimizer_fn, metric, image, label):
  """Perform one training step for the model.

  Args:
    model: Keras model to train.
    loss_fn: Loss function to use.
    optimizer_fn: Optimizer function to use.
    metric: keras.metric to use.
    image: Tensor of training images of shape [batch_size, 28, 28, 1].
    label: Tensor of class labels of shape [batch_size].
  """
  with tf.GradientTape() as tape:
    preds = model(image)
    label_onehot = tf.one_hot(label, 10)
    loss_ = loss_fn(label_onehot, preds)
  grads = tape.gradient(loss_, model.trainable_variables)
  optimizer_fn.apply_gradients(zip(grads, model.trainable_variables))
  metric(loss_)


def train_and_export(export_path,
                     buffer_size=1000,
                     batch_size=32,
                     learning_rate=1e-3,
                     epoch=10,
                     dataset=None):
  """Trains and export the model as SavedModel 2.0.

  Args:
    export_path: (str) Path to export the trained model.
    buffer_size: (int) Size of buffer to use while shuffling.
    batch_size: (int) Size of each training batch.
    learning_rate: (float) Learning rate to use for the optimizer.
    epoch: (int) Number of Epochs to train for.
    dataset: (tf.data.Dataset) Dataset object. Defaults to a MNIST dataset.
  """
  model = MNIST()
  if not dataset:
    dataset = tfds.load(
        "mnist", split="train", batch_size=batch_size).shuffle(
            buffer_size, reshuffle_each_iteration=True)

  optimizer_fn = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  metric = tf.keras.metrics.Mean()
  model.compile(optimizer_fn, loss=loss_fn)

  # Training loop.
  for epoch in range(epoch):
    for step, data in enumerate(dataset):
      train_step(model, loss_fn, optimizer_fn, metric, data["image"],
                 data["label"])
      print("\rEpoch: #{}\tStep: #{}\tLoss: {}\n".format(
          epoch, step,
          metric.result().numpy()))

  # We have to call either predict or fit to make it possible to export with
  # tf.keras.models.save_model.
  model.predict(next(iter(dataset))["image"])
  # Export the model as SavedModel 2.0.
  tf.saved_model.save(model, export_path)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--export_path",
      type=str,
      default=None,
      help="Path to export the module")
  parser.add_argument(
      "--buffer_size",
      type=int,
      default=1000,
      help="Buffer Size to use while shuffling the dataset")
  parser.add_argument(
      "--batch_size", type=int, default=32, help="Size of each batch")
  parser.add_argument(
      "--learning_rate", type=float, default=1e-3, help="learning rate")
  parser.add_argument(
      "--epoch", type=int, default=10, help="Number of iterations")
  FLAGS, unparsed = parser.parse_known_args()

  if not FLAGS.export_path:
    logging.error("Must set flag --export_path.")
    sys.exit(1)

  train_and_export(**vars(FLAGS))
