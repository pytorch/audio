#!/bin/bash
set -ex

echo "Running pre build script..."
export FFMPEG_ROOT=${PWD}/third_party/ffmpeg
if [[ ! -d ${FFMPEG_ROOT} ]]; then
    packaging/ffmpeg/build.sh
fi
echo FFMPEG_ROOT=${FFMPEG_ROOT}

echo "Setting environment variables for versions..."

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export BUILD_TYPE="conda"
export CU_VERSION="cu116"
setup_env
export SOURCE_ROOT_DIR="$PWD"
setup_conda_pytorch_constraint
setup_conda_cudatoolkit_constraint
setup_visual_studio_constraint

# nvidia channel included for cudatoolkit >= 11 however for 11.5 we use conda-forge
# HACK HACK HACK: Remove PYTHON_VERSION check once https://github.com/pytorch/builder/pull/961 is merged
export CUDATOOLKIT_CHANNEL="nvidia"
# NOTE: This is needed because `cudatoolkit=11.5` has a dependency on conda-forge
#       See: https://github.com/pytorch/audio/pull/2224#issuecomment-1049185550
export CUDA116_CUDA_DEPENDENCY=""
if [[ ${CU_VERSION} = "cu116" ]]; then
    export CUDATOOLKIT_CHANNEL="nvidia/label/cuda-11.6.2"
    export CUDA116_CUDA_DEPENDENCY="cuda"
elif [[ ! -z ${CU_VERSION} ]]; then
    export CUDA116_CUDA_DEPENDENCY="cudatoolkit"
fi

# NOTE: There are some dependencies that are not available for macOS on Python 3.10 without conda-forge
if [[ ${OSTYPE} =~ darwin* ]] && [[ ${PYTHON_VERSION} = "3.10" ]]; then
    CONDA_CHANNEL_FLAGS="${CONDA_CHANNEL_FLAGS} -c conda-forge"
fi
