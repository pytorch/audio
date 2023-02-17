#!/bin/bash
set -ex

echo FFMPEG_ROOT=${FFMPEG_ROOT}

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export BUILD_TYPE="conda"
setup_env
export SOURCE_ROOT_DIR="$PWD"
setup_conda_pytorch_constraint
setup_conda_cudatoolkit_constraint
setup_visual_studio_constraint

export CUDATOOLKIT_CHANNEL="nvidia"
# NOTE: There are some dependencies that are not available for macOS on Python 3.10 without conda-forge
if [[ ${OSTYPE} =~ darwin* ]] && [[ ${PYTHON_VERSION} = "3.10" ]]; then
    CONDA_CHANNEL_FLAGS="${CONDA_CHANNEL_FLAGS} -c conda-forge"
fi

if [[ "$PYTHON_VERSION" == "3.11" ]]; then
  export CONDA_CHANNEL_FLAGS="${CONDA_CHANNEL_FLAGS} -c malfet"
fi

conda build -c defaults -c $CUDATOOLKIT_CHANNEL ${CONDA_CHANNEL_FLAGS:-}  --no-anaconda-upload --no-test --python "$PYTHON_VERSION"  packaging/torchaudio
