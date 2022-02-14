#!/bin/bash
set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export BUILD_TYPE="conda"
setup_env 0.12.0
export SOURCE_ROOT_DIR="$PWD"
setup_conda_pytorch_constraint
setup_conda_cudatoolkit_constraint
setup_visual_studio_constraint

# nvidia channel included for cudatoolkit >= 11 however for 11.5 we use conda-forge
# HACK HACK HACK: Remove PYTHON_VERSION check once https://github.com/pytorch/builder/pull/961 is merged
export CUDATOOLKIT_CHANNEL="nvidia"
export ADDITIONAL_CHANNELS=""
if [[ "$CU_VERSION" == cu115 ]]; then
    export CUDATOOLKIT_CHANNEL="conda-forge"
fi
if [[ "$PYTHON_VERSION" == 3.10 ]]; then
    export ADDITIONAL_CHANNELS="-c conda-forge"
fi

conda build -c defaults -c $CUDATOOLKIT_CHANNEL $CONDA_CHANNEL_FLAGS $ADDITIONAL_CHANNELS --no-anaconda-upload --python "$PYTHON_VERSION" packaging/torchaudio
