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
# NOTE: This is needed because `cudatoolkit=11.5` has a dependency on conda-forge
#       See: https://github.com/pytorch/audio/pull/2224#issuecomment-1049185550
if [[ ${CU_VERSION} = "cu115" ]]; then
    CONDA_CHANNEL_FLAGS="${CONDA_CHANNEL_FLAGS} -c conda-forge"
fi
# NOTE: There are some dependencies that are not available for macOS on Python 3.10 without conda-forge
if [[ ${OSTYPE} =~ darwin* ]] && [[ ${PYTHON_VERSION} = "3.10" ]]; then
    CONDA_CHANNEL_FLAGS="${CONDA_CHANNEL_FLAGS} -c conda-forge"
fi
conda build -c defaults -c $CUDATOOLKIT_CHANNEL ${CONDA_CHANNEL_FLAGS:-} --no-anaconda-upload --python "$PYTHON_VERSION" packaging/torchaudio
