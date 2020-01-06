# Building an Approximate Nearest Neighbour Embedding Index for Similarity Matching

This `make_nearest_neighbour_index` tool helps you to generate embeddings from a
TF-Hub module given your text input data and build an approximate nearest
neighbours (ANN) index using the embeddings. The index can then be used for
real-time similarity matching and retrieval.

We use [Apache Beam](https://beam.apache.org/documentation/programming-guide/)
to generate the embeddings from the TF-Hub module.
We also use Spotify's [ANNOY](https://github.com/spotify/annoy) library to
build the approximate nearest neighbours index.

This tool uses **TensorFlow 2.0**.


## Tool setup
In order for you to use the tool in your local machine, you need to perform the
following steps:

```
$ pip install "tensorflow~=2.0"
$ pip install "tensorflow-hub[make_nearest_neighbour_index]~=0.8"
```

After installation, the `make_nearest_neighbour_index` executable is available
on the commandline:

```
$ make_nearest_neighbour_index --help
```

## Tool usage
The  make_nearest_neighbour_index expects one of the following four commands:

### 1- generate
The **generate** command generates embeddings for text input data using a TF-Hub
module. The following are the parameters expected by the command:

Parameter              | Type    | Description  |
---------------------- |---------| -------------|
 data_file_pattern     | string  | Path to data file(s) to generate embeddings for. The data is expected to be a single-column TSV.|
 module_url            | string  | TF-Hub module to use. For more options, search https://tfhub.dev. This also can be a path to a [saved_model](https://www.tensorflow.org/guide/saved_model) directory|
 embed_output_dir      | string  | The directory to store the generated embedding files to.|
 projected_dim         | int     | **(Optional)** The desired target dimension to project the embedding to. If specified, [random projection](https://en.wikipedia.org/wiki/Random_projection) will be uses. |

The following is an example usage of the command. The command generates text
embeddings for a set of titles in titles-\*.txt input files using the tf2
[nnlm-en-128](https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1)
TF-Hub-module. In addition, it performs random projection of the generated
embeddings to reduce the dimensionality from 128 to 64 (project-dim).

```
make_nearest_neighbour_index generate \
	--data_file_pattern=./data/titles-*.txt \
	--module_url=https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1 \
	--embed_output_dir=./output/embed/ \
	--projected_dim=64
```

This command produces (one or several) **.tfrecords** embedding files to the
**embed_output_dir** location. In addition, if random projection was performed,
a **random_projection.matrix** file is also produced in the **embed_output_dir**
location, which is a pickled numpy array of the projection weights.
This is needed for projected the input query and searching the embedding index.

### 2- build
The **build** command build an ANN index for input embeddings.
The following are the parameters expected by the command:

Parameter              | Type    | Description  |
---------------------- |---------| -------------|
 embed_output_dir    | string  | The directory of the .tfrecords file(s) with the embeddings to build the ANN index for.|
 num_trees             | int     | **(Optional)** The number of trees to build the ANN index. For more details, refer to https://github.com/spotify/annoy. **Default is 100.** |
 index_output_dir      | string  | The directory to store the created index and mapping files. |

The following is an example usage of the command. The command builds an ANN
index with 10 trees for embeddings in .tfrecord files with 64 dimensions.

```
make_nearest_neighbour_index build \
	--embed_output_dir='./embed/ \
	--index_output_dir=./output/index/ \
	--num_trees=10
```

This command produces two files:

1. **ann.index**: The serialized ANN index for the embeddings.

2. **ann.index.mapping**: A pickled dictionary to map the internal index
identifier of an item  to the original item.

3. **random_projection.matrix**: If a random projection matrix was created in
the embedding generation step, it will be copied to the index output directory.

### 3- e2e
The **e2e** command performs both embedding generation and index building steps.
The following is an example usage of the command.

```
make_nearest_neighbour_index e2e \
	--data_file_pattern=./test_data/large.txt \
	--module_url=https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1 \
	--embed_output_dir=./output/embed/ \
	--index_output_dir=./output/index/ \
	--projected_dim=64 \
	--num_trees=100
```

### 4- query
The **query** command allows you to use an ANN index to find similar items to
a given one. The following are the parameters expected by the command:

Parameter              | Type    | Description  |
---------------------- |---------| -------------|
 module_url            | string  | TF-Hub module to use to generate embedding for the input query item. This must be the same module used to generate embeddings in the ANN index. |
 index_output_dir      | string  | A directory containing the **ann.index** and **ann.index.mapping** files. |
 num_matches           | int     | The number of similar items to retrieve from the inded. **Default is 5**|

```
make_nearest_neighbour_index query \
	--module_url=https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1 \
	--index_output_dir=./output/index \
  --num_matches=10
```

This command will load the provided ANN index, the random projection matrix
(if provided), and the TF-Hub module, then perform the following:

1.  Accept an input query item from commandline.

2.  Generate embedding for the input item using the TF-Hub module.

3.  (Optional) if a random projection matrix is provided, the embedding is
    projected to the reduced dimensionality using the matrix weights.

4.  The ANN index is queried using the input item embeddings to retrieve the
    identifiers of the similar items.

5.  The mapping is used to translate the ANN item identifier to the original
    item.

6.  The similar items are displayed.
