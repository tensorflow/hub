# Copyright 2018 The TensorFlow Hub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Setup for pip package."""

from datetime import datetime
from setuptools import find_packages
from setuptools import setup

import sys

# Can't import the module during setup.py.
# Use execfile to find __version__.
with open('tensorflow_hub/version.py') as in_file:
  exec(in_file.read())

REQUIRED_PACKAGES = [
    'numpy >= 1.12.0',
    'protobuf >= 3.19.6',  # No less than what ../WORKSPACE uses.
    'tf-keras >= 2.14.1',
]

project_name = 'tensorflow-hub'
if '--project_name' in sys.argv:
  project_name_idx = sys.argv.index('--project_name')
  project_name = sys.argv[project_name_idx + 1]
  sys.argv.remove('--project_name')
  sys.argv.pop(project_name_idx)

# If we're dealing with a nightly build we need to make sure that the
# version changes for every release.
version = __version__
if project_name == 'tf-hub-nightly':
  version += datetime.now().strftime('%Y%m%d%H%M')

setup(
    name=project_name,  # Automatic: tensorflow_hub, etc. Case insensitive.
    version=version.replace('-', ''),
    description=(
        'TensorFlow Hub is a library to foster the publication, '
        'discovery, and consumption of reusable parts of machine '
        'learning models.'
    ),
    long_description='',
    url='https://github.com/tensorflow/hub',
    author='Google LLC',
    author_email='packages@tensorflow.org',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    extras_require={},
    entry_points={},
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords=(
        'tensorflow machine learning share module subgraph component hub '
        'embedding retraining transfer'
    ),
)
