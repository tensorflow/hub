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
"""Tests for module search utility."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from tensorflow_hub.tools.module_search import utils

class TestUtils(tf.test.TestCase):

    train_samples = 450
    test_samples = 50
    dim = 10
    classes = 7
    random_seed = 127

    def test_compute_distance_matrix(self):
        np.random.seed(seed=self.random_seed)
        x_train = np.random.rand(self.train_samples, self.dim)
        x_test = np.random.rand(self.test_samples, self.dim)

        print(x_test)

        d = utils.compute_distance_matrix(x_train, x_test)
        self.assertEqual(d.shape, (self.test_samples, self.train_samples))

        for i in range(self.test_samples):
            for j in range(self.train_samples):
                d_ij = np.linalg.norm(x_train[j,:] - x_test[i,:])**2
                self.assertAlmostEqual(d_ij, d[i,j], places=5)

    def test_compute_distance_matrix_loo(self):
        np.random.seed(seed=self.random_seed)
        x_train = np.random.rand(self.train_samples, self.dim)

        d = utils.compute_distance_matrix_loo(x_train)
        self.assertEqual(d.shape, (self.train_samples, self.train_samples))

        for i in range(self.train_samples):
            for j in range(self.train_samples):
                if i == j:
                    self.assertEqual(float("inf"), d[i,j])
                else:
                    d_ij = np.linalg.norm(x_train[j,:] - x_train[i,:])**2
                    self.assertAlmostEqual(d_ij, d[i,j], places=5)

    def knn_errorrate(self, k):
        x_train = np.random.rand(self.train_samples, self.dim)
        x_test = np.random.rand(self.test_samples, self.dim)

        d = utils.compute_distance_matrix(x_train, x_test)

        y_test = np.random.randint(self.classes, size=self.test_samples)
        y_train = np.random.randint(self.classes, size=self.train_samples)

        err = utils.knn_errorrate(d, y_train, y_test, k=k)
        
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        
        self.assertAlmostEqual(1.0 - err, metrics.accuracy_score(y_test, y_pred), places=5)

    def test_knn_errorrate(self):
        np.random.seed(seed=self.random_seed)
        ks = [1,3,5]
        for i in range(len(ks)):
            with self.subTest(i=i):
                self.knn_errorrate(ks[i])

    def knn_errorrate_loo(self, k):
        x_train = np.random.rand(self.train_samples, self.dim)

        d = utils.compute_distance_matrix_loo(x_train)

        y_train = np.random.randint(self.classes, size=self.train_samples)

        err = utils.knn_errorrate_loo(d, y_train, k=k)

        cnt = 0.0
        for i in range(self.train_samples):
            knn = KNeighborsClassifier(n_neighbors=k)
            mask = [True if j != i else False for j in range(self.train_samples)]
            knn.fit(x_train[mask], y_train[mask])
            y_pred = knn.predict(x_train[i].reshape(-1, self.dim))
            if y_pred != y_train[i]:
                cnt += 1
        
        self.assertAlmostEqual(err, cnt / self.train_samples, places=5)

    def test_knn_errorrate_loo(self):
        np.random.seed(seed=self.random_seed)
        ks = [1,3,5]
        for i in range(len(ks)):
            with self.subTest(i=i):
                self.knn_errorrate_loo(ks[i])

if __name__ == '__main__':
    tf.test.main()
