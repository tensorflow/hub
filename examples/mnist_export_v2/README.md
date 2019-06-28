MNIST classifier to export as Tensorflow Hub Module
------------------------------------------------------
### To export
The script expects that TensorFlow 2.0 is installed, e.g.

```bash
pip install tensorflow==2.0.0-beta0
```
Run the exporter as:

```bash
python3 export.py --export_path=/tmp/mnist_module
```
### To run the tests
From the project root directory run:

```
bazel test examples/mnist_export_v2:export_test
```
