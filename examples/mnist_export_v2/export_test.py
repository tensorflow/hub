import tensorflow as tf
import tensorflow_hub as hub
import unittest
import tempfile
import os
import re
import export


class TFHubMNISTTest(tf.test.TestCase):
    def setUp(self):
        self.mock_dataset = tf.data.Dataset.range(5)
        self.mock_dataset = self.mock_dataset.map(
            lambda x: {
                "image": tf.cast(
                    255 * tf.random.normal([1, 28, 28, 1]), tf.uint8),
                "label": x})

    def test_model_exporting(self):
        export.train_and_export(
            epoch=1,
            dataset=self.mock_dataset,
            export_path="%s/model/1" %
            self.get_temp_dir())
        self.assertTrue(os.listdir(self.get_temp_dir()))

    def test_empty_input(self):
        if not os.path.exists("%s/model/1" % self.get_temp_dir()):
            export.train_and_export(
                epoch=1,
                dataset=self.mock_dataset,
                export_path="%s/model/1" %
                self.get_temp_dir())
        model = hub.load("%s/model/1" % self.get_temp_dir())
        output_ = model.call(
            tf.zeros([1, 28, 28, 1], dtype=tf.uint8).numpy())
        self.assertEqual(output_.shape, [1, 10])


if __name__ == '__main__':
    tf.test.main()
