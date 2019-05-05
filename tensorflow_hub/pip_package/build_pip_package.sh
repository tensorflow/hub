#!/usr/bin/env bash
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

# This script should be run from the repo root.

set -e
set -o pipefail

die() {
  printf >&2 '%s\n' "$1"
  exit 1
}

function main() {
  if [ $# -lt 1 ] ; then
    die "ERROR: no destination dir provided"
  fi

  DEST=$1
  TMPDIR=$(mktemp -d)
  RUNFILES="bazel-bin/tensorflow_hub/pip_package/build_pip_package.runfiles/org_tensorflow_hub"

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"

  bazel build //tensorflow_hub/pip_package:build_pip_package

  if [ ! -d bazel-bin/tensorflow_hub ]; then
    echo `pwd`
    die "ERROR: Could not find bazel-bin.  Did you run from the build root?"
  fi

  cp "tensorflow_hub/pip_package/setup.py" "${TMPDIR}"
  cp "tensorflow_hub/pip_package/setup.cfg" "${TMPDIR}"
  cp "LICENSE" "${TMPDIR}/LICENSE.txt"
  cp -R "${RUNFILES}/tensorflow_hub" "${TMPDIR}"

  pushd ${TMPDIR}
  rm -f MANIFEST

  echo $(date) : "=== Building universal python wheel in $PWD"
  python setup.py bdist_wheel --universal >/dev/null
  mkdir -p ${DEST}
  cp dist/* ${DEST}
  popd
  rm -rf ${TMPDIR}
  echo $(date) : "=== Output wheel files are in: ${DEST}"
}

main "$@"
