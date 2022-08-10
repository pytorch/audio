#!/bin/bash
set -ex

echo "Running pre build script..."
export FFMPEG_ROOT=${PWD}/third_party/ffmpeg
if [[ ! -d ${FFMPEG_ROOT} ]]; then
    packaging/ffmpeg/build.sh
fi
echo FFMPEG_ROOT=${FFMPEG_ROOT}

echo "Setting environment variables for versions..."
export PYTHON_VERSION='3.7'
export CU_VERSION=cu113


script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export BUILD_TYPE="wheel"
setup_env
setup_wheel_python
pip_install numpy future cmake ninja
setup_pip_pytorch_version
