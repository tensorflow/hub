# Examples

## Notebooks

#### [`colab/text_classification_with_tf_hub_on_kaggle.ipynb`](colab/text_classification_with_tf_hub_on_kaggle.ipynb)

Shows how to solve a problem on Kaggle with TF-Hub.

#### [`colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb`](colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb)

Explores text semantic similarity with the
[Universal Encoder Module](https://tfhub.dev/google/universal-sentence-encoder/2).

#### [`colab/tf_hub_generative_image_module.ipynb`](colab/tf_hub_generative_image_module.ipynb)

Explores a generative image module.

#### [`colab/action_recognition_with_tf_hub.ipynb`](colab/action_recognition_with_tf_hub.ipynb)

Explores action recognition from video.

#### [`colab/tf_hub_delf_module.ipynb`](colab/tf_hub_delf_module.ipynb)

Exemplifies use of the [DELF Module](https://tfhub.dev/google/delf/1) for
landmark recognition and matching.

#### [`colab/object_detection.ipynb`](colab/object_detection.ipynb)

Explores object detection with the use of the
[Faster R-CNN module trained on Open Images v4](https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1).

#### [`colab/tf2_semantic_approximate_nearest_neighbors`](colab/tf2_semantic_approximate_nearest_neighbors.ipynb)

This tutorial illustrates how to generate embeddings from a
[TF2 SavedModel](https://www.tensorflow.org/hub/tf2_saved_model) given input
data and build an approximate nearest neighbours (ANN) index using the extracted
embeddings for real-time similarity matching and retrieval.

#### [`colab/semantic_approximate_nearest_neighbors`](colab/semantic_approximate_nearest_neighbors.ipynb)

This tutorial illustrates how to generate embeddings from a model in the legacy
[TF1 Hub format](https://www.tensorflow.org/hub/tf1_hub_module) given
input data and build an approximate nearest neighbours (ANN) index using the
extracted embeddings for real-time similarity matching and retrieval.

## Python scripts

#### [`image_retraining`](image_retraining)

Shows how to train an image classifier based on any TensorFlow Hub module that
computes image feature vectors.

#### [`text_embeddings`](text_embeddings)

Example tool to generate a text embedding module from a csv file with word
embeddings.

#### [`half_plus_two`](half_plus_two)

Simple example of how to create a TensorFlow Hub Module.

### TensorFlow 2

#### [`text_embeddings_v2`](text_embeddings)

Example tool to generate a text embedding module in TF2 format.

#### [`mnist`](mnist_export_v2)

Example tool to train and export a simple MNIST classifier in TF2 format.
