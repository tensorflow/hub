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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.python.tpu import tpu_estimator
from tensorflow.python.tpu import tpu_optimizer
from tensorflow.python.tpu import tpu_config
from absl import flags, app

flags.DEFINE_string("tpu", None, "TPU Address")
flags.DEFINE_integer("iterations", 2, "Number of Itertions")
flags.DEFINE_integer("batch_size", 16, "Size of each Batch")
flags.DEFINE_float("learning_rate", 1e-3, "Learning Rate")
flags.DEFINE_boolean("use_tpu", True, " Use TPU")
flags.DEFINE_boolean("use_compat", True, "Use OptimizerV1 from compat module")
flags.DEFINE_integer(
    "max_steps",
    1000,
    "Maximum Number of Steps for TPU Estimator")
flags.DEFINE_string(
    "model_dir",
    "model_dir/",
    "Directory to Save the Models and Checkpoint")
flags.DEFINE_string(
    "dataset",
    "horses_or_humans",
    "TFDS Dataset Name. IMAGE Dimension should be >= 224, channel=3")
flags.DEFINE_string("data_dir", None, "Directory to Save Data to")
flags.DEFINE_string("infer", None, "Dummy image file to infer")

FLAGS = flags.FLAGS
NUM_CLASSES = None


def resize_and_scale(image, label):
  image = tf.image.resize(image, size=[224, 224])
  image = tf.cast(image, tf.float32)
  image = image / tf.reduce_max(tf.gather(image, 0))
  return image, label


def input_(mode, batch_size, iterations, **kwargs):
  global NUM_CLASSES
  dataset, info = tfds.load(
      kwargs["dataset"],
      as_supervised=True,
      split="train" if mode == tf.estimator.ModeKeys.TRAIN else "test",
      with_info=True,
      data_dir=kwargs['data_dir']
  )
  NUM_CLASSES = info.features['label'].num_classes
  dataset = dataset.map(resize_and_scale).shuffle(
      1000).repeat(iterations).batch(batch_size, drop_remainder=True)
  return dataset


def model_fn(features, labels, mode, params):
  global NUM_CLASSES
  assert NUM_CLASSES is not None
  model = tf.keras.Sequential([
      hub.KerasLayer("https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4",
                     output_shape=[2048],
                     trainable=False
                     ),
      tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
  ])
  optimizer = None
  if mode == tf.estimator.ModeKeys.TRAIN:
    if not params["use_compat"]:
      optimizer = tf.optimizers.Adam(params["learning_rate"])
    else:
      optimizer = tf.compat.v1.train.AdamOptimizer(
          params["learning_rate"])
    if params["use_tpu"]:
      optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

  with tf.GradientTape() as tape:
    logits = model(features)
    if mode == tf.estimator.ModeKeys.PREDICT:
      preds = {
          "predictions": logits
      }
      return tpu_estimator.TPUEstimatorSpec(mode, predictions=preds)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)(labels, logits)
  if mode == tf.estimator.ModeKeys.EVAL:
    return tpu_estimator.TPUEstimatorSpec(mode, loss=loss)

  def train_fn(use_compat):
    assert optimizer is not None
    gradient = tape.gradient(loss, model.trainable_variables)
    global_step = tf.compat.v1.train.get_global_step()
    apply_grads = tf.no_op()  # Does Nothing. Initialization only. None would also work
    if not use_compat:
      update_global_step = tf.compat.v1.assign(
          global_step, global_step + 1, name='update_global_step')
      with tf.control_dependencies([update_global_step]):
        apply_grads = optimizer.apply_gradients(
            zip(gradient, model.trainable_variables))
    else:
      apply_grads = optimizer.apply_gradients(
          zip(gradient, model.trainable_variables),
          global_step=global_step)
    return apply_grads

  if mode == tf.estimator.ModeKeys.TRAIN:
    return tpu_estimator.TPUEstimatorSpec(
        mode, loss=loss, train_op=train_fn(
            params['use_compat']))


def main(_):
  os.environ["TFHUB_CACHE_DIR"] = os.path.join(
      FLAGS.model_dir, "tfhub_modules")
  os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
  input_fn = partial(input_, iterations=FLAGS.iterations)
  cluster = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
  run_config = tpu_config.RunConfig(
      model_dir=FLAGS.model_dir,
      cluster=cluster,
      tpu_config=tpu_config.TPUConfig(FLAGS.iterations))

  classifier = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size,
      config=run_config,
      params={
          "use_tpu": FLAGS.use_tpu,
          "data_dir": FLAGS.data_dir,
          "dataset": FLAGS.dataset,
          "use_compat": FLAGS.use_compat,
          "learning_rate": FLAGS.learning_rate
      }
  )
  try:
    classifier.train(
        input_fn=lambda params: input_fn(
            mode=tf.estimator.ModeKeys.TRAIN,
            **params),
        max_steps=FLAGS.max_steps)
  except Exception:
    pass
  if FLAGS.infer:
    def prepare_input_fn(path):
      img = tf.image.decode_image(tf.io.read_file(path))
      return resize_and_scale(img, None)

    predictions = classifer.predict(
        input_fn=lambda params: prepare_input_fn(FLAGS.infer))
    print(predictions)


if __name__ == "__main__":
  app.run(main)
