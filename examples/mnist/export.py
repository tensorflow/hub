import sys
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import argparse
""" Create a Sample TF-Hub Module using SavedModel v2.0
The module has as a single signature which loads MNIST Dataset from TFDS and train a simple Neural Network for classifying the digits. The model is built and trained using Tewnsorlfow
"""
FLAGS = None


class MNIST(tf.keras.models.Model):
  """Model class for MNIST Classifier
  """

  def __init__(self, output_activation="softmax"):
    """
    Args:
      output_activation (str): activation for last layer
    """
    super(MNIST, self).__init__()
    self.layer_1 = tf.keras.layers.Dense(64)
    self.layer_2 = tf.keras.layers.Dense(10, activation=output_activation)

  @tf.function(
      input_signature=[
          tf.TensorSpec(
              shape=[
                  None,
                  28,
                  28,
                  1],
              dtype=tf.uint8)])
  def call(self, inputs):
    casted = tf.keras.layers.Lambda(
        lambda x: tf.cast(x, tf.float32))(inputs)
    flatten = tf.keras.layers.Flatten()(casted)
    normalize = tf.keras.layers.Lambda(
        lambda x: x / tf.reduce_max(tf.gather(x, 0)))(flatten)
    x = self.layer_1(normalize)
    output = self.layer_2(x)
    return output


def train_step(model, loss_fn, optimizer_fn, metric, image, label):
  """ Trains the model
    Args:
      model: Keras Model to train
      loss_fn: Loss Function to use
      optimizer_fn: Optimizer function to use
      metric: keras.metric to use
      image: shape[batch_size, width, height, num_channels] Tensor of Training Images
      label: shape[batch_size,] Image classes as returned by TFDS
  """
  with tf.GradientTape() as tape:
    preds = model(image)
    label_onehot = tf.one_hot(label, 10)
    loss_ = loss_fn(label_onehot, preds)
  grads = tape.gradient(loss_, model.trainable_variables)
  optimizer_fn.apply_gradients(zip(grads, model.trainable_variables))
  metric(loss_)


def train_and_export(
        data_dir=None,
        buffer_size=1000,
        batch_size=32,
        epoch=10,
        export_path="/tmp/tfhub_modules/mnist/digits/1"):
  """
    Trains and export the Model as SavedModel 2.0
    Args:
      data_dir (str): Directory where to store datasets from TFDS (With proper authentication, Cloud Bucket Supported)
      buffer_size (int): Size of Buffer to use while shuffling
      batch_size (int): size of each training batch
      export_path (str): path to export the trained model
  """
  model = MNIST()
  kwargs = {}
  if data_dir:
    kwargs["data_dir"] = data_dir
  train = tfds.load("mnist", split="train", **kwargs)
  optimizer_fn = tf.optimizers.Adam(learning_rate=1e-3)
  loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  metric = tf.keras.metrics.Mean()
  model.compile(optimizer_fn, loss=loss_fn)
  train = train.shuffle(
      buffer_size,
      reshuffle_each_iteration=True).batch(
      batch_size)
  # Training Loop
  for epoch in range(epoch):
    for step, data in enumerate(train):
      train_step(
          model,
          loss_fn,
          optimizer_fn,
          metric,
          data['image'],
          data['label'])
      sys.stdout.write("\rEpoch: #{}\tStep: #{}\tLoss: {}".format(
          epoch, step, metric.result().numpy()))
  # Exporting Model as SavedModel 2.0
  tf.saved_model.save(model, export_path)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--export_path",
      type=str,
      default="/tmp/tfhub_modules/mnist/digits/1",
      help="Path to export the module")
  parser.add_argument("--data_dir", type=str, default=None,
                      help="Path to Custom TFDS Data Directory")
  parser.add_argument(
      "--buffer_size",
      type=int,
      default=1000,
      help="Buffer Size to Use while Shuffling the Dataset")
  parser.add_argument(
      "--batch_size",
      type=int,
      default=32,
      help="Size of each batch")
  parser.add_argument(
      "--epoch",
      type=int,
      default=10,
      help="Number of iterations")
  FLAGS, unparsed = parser.parse_known_args()
  train_and_export(**vars(FLAGS))
