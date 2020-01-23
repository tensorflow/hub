# Module google/text-embedding-model/1
Token based text embedding trained on English Wikipedia corpus[1].

<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- module-type: text-embedding -->
<!-- network-architecture: word2vec -->
<!-- network-architecture: skip-gram -->
<!-- dataset: Wikipedia -->
<!-- language: en -->
<!-- fine-tunable: true -->
<!-- format: saved_model_2 -->

## Overview

Text embedding based on skipgram version of word2vec with 1 out-of-vocabulary
bucket. Maps from text to 500-dimensional embedding vectors.

#### Example use
The saved model can be loaded directly:

```
import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/text-embedding-model/1")
embeddings = embed(["cat is on the mat", "dog is in the fog"])
```

It can also be used within Keras:

```
hub_layer = hub.KerasLayer("https://tfhub.dev/google/text-embedding-model/1",
                           input_shape=[], dtype=tf.string)

model = keras.Sequential()
model.add(hub_layer)
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()
```

#### References
[1] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean.
[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781).
In Proceedings of Workshop at ICLR, 2013.
