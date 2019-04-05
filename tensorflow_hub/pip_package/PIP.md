<!-- Copyright 2018 The TensorFlow Hub Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=============================================================================-->
# Creating the TensorFlow Hub pip package using Linux

This requires Python, Bazel (0.18.1 or Similar) and Git. (And TensorFlow for testing the package.)

### Activate virtualenv

Install virtualenv if it's not installed already:

```shell
~$ sudo apt-get install python-virtualenv
```

Create a virtual environment for the package creation:

```shell
~$ virtualenv --system-site-packages tensorflow_hub_env
```

And activate it:

```shell
~$ source ~/tensorflow_hub_env/bin/activate  # bash, sh, ksh, or zsh
~$ source ~/tensorflow_hub_env/bin/activate.csh  # csh or tcsh
```

### Clone the TensorFlow Hub repository.

```shell
(tensorflow_hub_env)~/$ git clone https://github.com/tensorflow/hub
```

### Build TensorFlow Hub pip packaging script

To build a pip package for TensorFlow Hub:

```shell
(tensorflow_hub_env)~/$ bazel build tensorflow_hub/pip_package:build_pip_package
```

### Create the TensorFlow Hub pip package

```shell
(tensorflow_hub_env)~/$ bazel-bin/tensorflow_hub/pip_package/build_pip_package \
/tmp/tensorflow_hub_pkg
```

### Install and test the pip package (optional)

Run the following commands to install the pip package and test TensorFlow Hub.

```shell
(tensorflow_hub_env)~/$ pip install /tmp/tensorflow_hub_pkg/*.whl
(tensorflow_hub_env)~/$ python -c "import tensorflow_hub as hub"
```

### De-activate the virtualenv

```shell
(tensorflow_hub_env)~/$ deactivate
```
