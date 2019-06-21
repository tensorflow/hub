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
        "//tensorflow_hub:expect_numpy_installed",
        "//tensorflow_hub:expect_tensorflow_installed",
        "//tensorflow_hub",
    ],
)

py_test(
    name = "export_test",
    srcs = ["export_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":export_lib",
        "//tensorflow_hub:expect_numpy_installed",
        "//tensorflow_hub:expect_tensorflow_installed",
        "//tensorflow_hub",
    ],
)
