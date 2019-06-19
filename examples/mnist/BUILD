py_binary(
    name = "export",
    srcs = ["export.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [":export_lib"],
)

py_library(
    name = "export_lib",
    srcs = ["export.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":expect_numpy_installed",
        "//external:tensorflow_hub",
    ],
)

py_library(
    name = "expect_numpy_installed",
)

py_library(
    name = "expect_tensorflow_installed",
)

py_test(
    name = "export_test",
    srcs = ["export_test.py"],
    data = glob(["test_data/*.jpg"]),
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":expect_numpy_installed",
        ":expect_tensorflow_installed",
        ":export_lib",
        "//external:tensorflow_hub",
    ],
)
