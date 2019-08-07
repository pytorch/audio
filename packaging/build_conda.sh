#!/bin/bash
set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

setup_build_version 0.4.0
setup_macos
export SOURCE_ROOT_DIR="$PWD"
setup_conda_pytorch_constraint
conda build $CONDA_CHANNEL_FLAGS --no-anaconda-upload --python "$PYTHON_VERSION" packaging/torchaudio

