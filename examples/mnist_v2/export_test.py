import tensorflow as tf
import tensorflow_hub as hub
import unittest
import tempfile
import os
from absl.testing import absltest
import export

model = None
TEST_DATA_DIR = "examples/mnist_v2/test_data"

def load_model(path):
  global model
  if not model:
    model = hub.load(path)
  return model


class TFHubMNISTTest(tf.test.TestCase):
  def setUp(self):
    file_ = tf.gather(tf.io.matching_files("%s/*.jpg" % TEST_DATA_DIR), 0)
    self.test_image = tf.expand_dims(
        tf.image.decode_jpeg(
            tf.io.read_file(file_)), 0)
    self.test_label = int(str(file_.numpy()).split("/")[-1].split(".jpg")[0])

  def test_model_exporting(self):
    mocked_dict = export.sys.__dict__.copy()
    with open(os.devnull, "w") as devnull:
      mocked_dict["stdout"] = devnull
      absltest.mock.patch.dict(
          export.sys.__dict__,
          mocked_dict,
          clear=True).start()
      export.train_and_export(epoch=1, export_path="%s/model/1" % self.get_temp_dir())
      self.assertTrue(os.listdir(self.get_temp_dir()))

  def test_empty_input(self):
    if not os.path.exists("%s/model/1" % self.get_temp_dir()):
      self.test_model_exporting()
    model = load_model("%s/model/1" % self.get_temp_dir())
    output_ = model.call(
        tf.zeros([1, 28, 28, 1], dtype=tf.uint8).numpy())
    self.assertEqual(output_.shape, [1, 10])

  def test_sample_input(self):
    if not os.path.exists(os.path.join(self.get_temp_dir(), "model/1")):
      self.test_model_exporting()
    model = load_model("%s/model/1" % self.get_temp_dir())
    prediction = model.call(self.test_image)
    self.assertEqual(
        tf.argmax(
            tf.squeeze(prediction)).numpy(),
        self.test_label)

if __name__ == '__main__':
  tf.test.main()
