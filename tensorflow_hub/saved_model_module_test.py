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


def _double(input_):
  return input_ * 2


class MyModel(tf.Module):

  @tf.function
  def __call__(self, input_):
    return _double(input_)


class SavedModelTest(tf.test.TestCase):

  def _create_tf2_saved_model(self):
    model_dir = os.path.join(self.get_temp_dir(), "saved_model")
    model = MyModel()

    @tf.function
    def serving_default(input_):
      return {"output": model(input_)}

    signature_function = serving_default.get_concrete_function(
        tf.TensorSpec(shape=[3,], dtype=tf.float32)
    )
    tf.saved_model.save(
        model,
        model_dir,
        signatures={
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_function
        },
    )

    return model_dir

  def testLoadSavedModel(self):
    saved_model_path = self._create_tf2_saved_model()
    loaded = hub.load(saved_model_path)
    self.assertAllClose(
        loaded.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY](
            tf.constant([2.0, 4.0, 5.0])
        )["output"],
        tf.constant([4.0, 8.0, 10.0]),
    )


if __name__ == "__main__":
  tf.test.main()
