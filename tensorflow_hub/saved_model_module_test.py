# Copyright 2018 The TensorFlow Hub Authors. All Rights Reserved.
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
"""Tests for tensorflow_hub.saved_model."""

import os

import tensorflow as tf
import tensorflow_hub as hub


_EXTRA_COLLECTION = "exercise_drop_collection"


class SavedModelTest(tf.test.TestCase):

  def createSavedModel(self):
    model_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with tf.Graph().as_default():
      x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 3])
      w = tf.compat.v1.get_variable("weights", shape=[])
      y = x*w
      tf.compat.v1.add_to_collection(_EXTRA_COLLECTION, y)

      init_op = tf.compat.v1.assign(w, 2)

      with tf.compat.v1.Session() as session:
        session.run(init_op)
        tf.compat.v1.saved_model.simple_save(
            session,
            model_dir,
            inputs={"x": x},
            outputs={"y": y},
        )
    return model_dir

  def testLoadSavedModel(self):
    saved_model_path = self.createSavedModel()
    spec = hub.create_module_spec_from_saved_model(
        saved_model_path,
        drop_collections=[_EXTRA_COLLECTION])
    with tf.Graph().as_default():
      m = hub.Module(spec, tags=["serve"])
      y = m([[2, 4, 5]], signature="serving_default", as_dict=True)["y"]
      with tf.compat.v1.train.MonitoredSession() as session:
        self.assertAllEqual(session.run(y), [[4, 8, 10]])


if __name__ == "__main__":
  tf.test.main()
