WARNING: This code is deprecated.

The `retrain.py` tool from this directory has been replaced by the
[`make_image_classifier`](https://github.com/tensorflow/hub/tree/master/tensorflow_hub/tools/make_image_classifier)
tool that gets
[installed](https://www.tensorflow.org/hub/installation) as a command-line tool
by the PIP package `tensorflow-hub[make_image_classifier]`.
The new tool uses TensorFlow 2 and supports fine-tuning.

The Colab notebook
[tf2_image_retraining.ipynb](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb)
explains the basic technique behind that tool: transfer learning with TF Hub.
