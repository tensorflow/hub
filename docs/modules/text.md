# Text Modules

Modules pre-trained to embed words, phrases, and sentences as many-dimensional
vectors.

Click on a module to view its documentation, or reference the URL from the
TensorFlow Hub library like so:

```python
m = hub.Module("https://tfhub.dev/...")
```

### Universal Sentence Encoder
Encoder of greater-than-word length text trained on a variety of data.

* [universal-sentence-encoder](https://tfhub.dev/google/universal-sentence-encoder/2)
* [universal-sentence-encoder-large](https://tfhub.dev/google/universal-sentence-encoder-large/3)
* [universal-sentence-encoder-lite](https://tfhub.dev/google/universal-sentence-encoder-lite/2) (\*Text preprocessing required)

### ELMo
Deep Contextualized Word Representations trained on the 1 Billion Word Benchmark.

* [elmo](https://tfhub.dev/google/elmo/2)

### NNLM embedding trained on Google News
Embedding from a neural network language model trained on Google News dataset.

|            | 50 dimensions | 128 dimensions |
|------------|---------------|----------------|
| Chinese    | [nnlm-zh-dim50](https://tfhub.dev/google/nnlm-zh-dim50/1) <br/> [nnlm-zh-dim50-with-normalization](https://tfhub.dev/google/nnlm-zh-dim50-with-normalization/1) | [nnlm-zh-dim128](https://tfhub.dev/google/nnlm-zh-dim128/1) <br/> [nnlm-zh-dim128-with-normalization](https://tfhub.dev/google/nnlm-zh-dim128-with-normalization/1)
| English    | [nnlm-en-dim50](https://tfhub.dev/google/nnlm-en-dim50/1) <br/> [nnlm-en-dim50-with-normalization](https://tfhub.dev/google/nnlm-en-dim50-with-normalization/1) | [nnlm-en-dim128](https://tfhub.dev/google/nnlm-en-dim128/1) <br/> [nnlm-en-dim128-with-normalization](https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1)
| German     | [nnlm-de-dim50](https://tfhub.dev/google/nnlm-de-dim50/1) <br/> [nnlm-de-dim50-with-normalization](https://tfhub.dev/google/nnlm-de-dim50-with-normalization/1) | [nnlm-de-dim128](https://tfhub.dev/google/nnlm-de-dim128/1) <br/> [nnlm-de-dim128-with-normalization](https://tfhub.dev/google/nnlm-de-dim128-with-normalization/1)
| Indonesian | [nnlm-id-dim50](https://tfhub.dev/google/nnlm-id-dim50/1) <br/> [nnlm-id-dim50-with-normalization](https://tfhub.dev/google/nnlm-id-dim50-with-normalization/1) | [nnlm-id-dim128](https://tfhub.dev/google/nnlm-id-dim128/1) <br/> [nnlm-id-dim128-with-normalization](https://tfhub.dev/google/nnlm-id-dim128-with-normalization/1)
| Japanese   | [nnlm-ja-dim50](https://tfhub.dev/google/nnlm-ja-dim50/1) <br/> [nnlm-ja-dim50-with-normalization](https://tfhub.dev/google/nnlm-ja-dim50-with-normalization/1) | [nnlm-ja-dim128](https://tfhub.dev/google/nnlm-ja-dim128/1) <br/> [nnlm-ja-dim128-with-normalization](https://tfhub.dev/google/nnlm-ja-dim128-with-normalization/1)
| Korean     | [nnlm-ko-dim50](https://tfhub.dev/google/nnlm-ko-dim50/1) <br/> [nnlm-ko-dim50-with-normalization](https://tfhub.dev/google/nnlm-ko-dim50-with-normalization/1) | [nnlm-ko-dim128](https://tfhub.dev/google/nnlm-ko-dim128/1) <br/> [nnlm-ko-dim128-with-normalization](https://tfhub.dev/google/nnlm-ko-dim128-with-normalization/1)
| Spanish    | [nnlm-es-dim50](https://tfhub.dev/google/nnlm-es-dim50/1) <br/> [nnlm-es-dim50-with-normalization](https://tfhub.dev/google/nnlm-es-dim50-with-normalization/1) | [nnlm-es-dim128](https://tfhub.dev/google/nnlm-es-dim128/1) <br/> [nnlm-es-dim128-with-normalization](https://tfhub.dev/google/nnlm-es-dim128-with-normalization/1)

### Word2vec trained on Wikipedia
Embedding trained by word2vec on Wikipedia.

#### English
| 250 dimensions | 500 dimensions |
|----------------|----------------|
| [Wiki-words-250](https://tfhub.dev/google/Wiki-words-250/1) <br/> [Wiki-words-250-with-normalization](https://tfhub.dev/google/Wiki-words-250-with-normalization/1) | [Wiki-words-500](https://tfhub.dev/google/Wiki-words-500/1) <br/> [Wiki-words-500-with-normalization](https://tfhub.dev/google/Wiki-words-500-with-normalization/1)
