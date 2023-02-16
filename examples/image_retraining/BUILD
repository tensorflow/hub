# Description:
# Transfer learning example for TensorFlow.

package(default_applicable_licenses = ["//tensorflow_hub:license"])

licenses(["notice"])

py_binary(
    name = "retrain",
    srcs = [
        "retrain.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_hub:expect_numpy_installed",
        "//tensorflow_hub:expect_tensorflow_installed",
        "//tensorflow_hub",
        #  "tensorflow/contrib/quantize:quantize_graph",
    ],
)
