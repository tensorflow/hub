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
"""Utils for module search functionality."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds


def compute_distance_matrix(x_train, x_test, measure="squared_l2"):
  """Calculates the distance matrix between test and train.

  Args:
    x_train: Matrix (NxD) where each row represents a training sample
    x_test: Matrix (MxD) where each row represents a test sample
    measure: Distance measure (not necessarly metric) to use

  Raises:
    NotImplementedError: When the measure is not implemented

  Returns:
    Matrix (MxN) where elemnt i,j is the distance between
    x_test_i and x_train_j.
  """

  x_train = tf.convert_to_tensor(x_train, tf.float64)
  x_test = tf.convert_to_tensor(x_test, tf.float64)

  if measure == "squared_l2":
    x_xt = tf.matmul(x_test, tf.transpose(x_train)).numpy()

    x_train_2 = tf.reduce_sum(tf.math.square(x_train), 1).numpy()
    x_test_2 = tf.reduce_sum(tf.math.square(x_test), 1).numpy()

    for i in range(np.shape(x_xt)[0]):
      x_xt[i, :] = np.multiply(x_xt[i, :], -2)
      x_xt[i, :] = np.add(x_xt[i, :], x_test_2[i])
      x_xt[i, :] = np.add(x_xt[i, :], x_train_2)

  else:
    raise NotImplementedError("Method '{}' is not implemented".format(measure))

  return x_xt


def compute_distance_matrix_loo(x, measure="squared_l2"):
  """Calculates the distance matrix for leave-one-out strategy.

  Args:
    x: Matrix (NxD) where each row represents a sample
    measure: Distance measure (not necessarly metric) to use

  Raises:
    NotImplementedError: When the measure is not implemented

  Returns:
    Matrix (NxN) where elemnt i,j is the distance between x_i and x_j.
    The diagonal is set to infinity
  """

  x = tf.convert_to_tensor(x, tf.float64)

  if measure == "squared_l2":
    x_xt = tf.matmul(x, tf.transpose(x)).numpy()
    diag = np.diag(x_xt)
    d = np.copy(x_xt)

    for i in range(np.shape(d)[0]):
      d[i, :] = np.multiply(d[i, :], -2)
      d[i, :] = np.add(d[i, :], x_xt[i, i])
      d[i, :] = np.add(d[i, :], diag)
      d[i, i] = float("inf")

  elif measure == "cosine":
    d = tf.matmul(x, tf.transpose(x)).numpy()
    diag_sqrt = np.sqrt(np.diag(d))
    outer = np.outer(diag_sqrt, diag_sqrt)
    d = np.ones(np.shape(d)) - np.divide(d, outer)
    np.fill_diagonal(d, float("inf"))

  else:
    raise NotImplementedError("Method '{}' is not implemented".format(measure))

  return d


def knn_errorrate(d, y_train, y_test, k=1):
  """Calculate the knn error rate based on the distance matrix d.

  Args:
    d: distance matrix
    y_train: label vector for the training samples
    y_test: label vector for the test samples
    k: number of direct neighbors for knn

  Returns:
    knn error rate (1 - accuracy)
  """

  if k == 1:
    indices = np.argmin(d, axis=1)

    cnt = 0
    for i in range(len(indices)):
      if y_test[i] != y_train[indices[i]]:
        cnt += 1

    return float(cnt) / len(indices)

  indices = np.argpartition(d, k - 1, axis=1)
  cnt = 0
  for i in range(np.shape(d)[0]):
    cnt_i = 0
    for j in range(k):
      if y_test[i] != y_train[indices[i, j]]:
        cnt_i += 1
    if cnt_i >= k / 2.0:
      cnt += 1

  return float(cnt) / np.shape(d)[0]


def knn_errorrate_loo(d, y, k=1):
  """Calculate the leave-one-out expected knn error rate based
  on the distance matrix d.

  Args:
    d: distance matrix, the diagonal should be infinity
    y: label matrix
    k: number of direct neighbors for knn

  Returns:
    Expected leave-one-out knn error rate (1 - accuracy)
  """

  if k == 1:
    indices = np.argmin(d, axis=1)

    cnt = 0
    for i in range(len(indices)):
      if y[i] != y[indices[i]]:
        cnt += 1

    return float(cnt) / len(indices)

  indices = np.argpartition(d, k - 1, axis=1)
  cnt = 0
  for i in range(np.shape(d)[0]):
    cnt_i = 0
    for j in range(k):
      if y[i] != y[indices[i, j]]:
        cnt_i += 1
    if cnt_i >= k / 2.0:
      cnt += 1

  return float(cnt) / np.shape(d)[0]


def load_data(dataset, split, num_examples=None):
  ds = tfds.load(dataset, split=split, shuffle_files=False)
  if num_examples:
    ds = ds.take(num_examples)
  return ds


def load_embedding_fn(module):
  m = hub.load(module, tags=[])
  return lambda x: m.signatures["default"](x)["default"]
