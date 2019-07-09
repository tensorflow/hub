# Transfer learning example using TensorFlow 2.0 and Cloud TPUs.

licenses(["notice"])

py_binary(
    name = "tpu_retrain",
    srcs = [
        "image_retraining_tpu.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_hub:expect_numpy_installed",
        "//tensorflow_hub:expect_tensorflow_installed",
        "//tensorflow_hub:expect_tensorflow_datasets_installed",
        "//tensorflow_hub",
    ],
)
