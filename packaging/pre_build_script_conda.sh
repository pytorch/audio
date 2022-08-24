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
export CU_VERSION="cpu"
export PYTHON_VERSION="3.8"
setup_env
export SOURCE_ROOT_DIR="$PWD"
setup_conda_pytorch_constraint
setup_conda_cudatoolkit_constraint
setup_visual_studio_constraint

export CUDATOOLKIT_CHANNEL="nvidia"
# NOTE: There are some dependencies that are not available for macOS on Python 3.10 without conda-forge
# nit
if [[ ${OSTYPE} =~ darwin* ]] && [[ ${PYTHON_VERSION} = "3.10" ]]; then
    CONDA_CHANNEL_FLAGS="${CONDA_CHANNEL_FLAGS} -c conda-forge"
fi
