#!/bin/bash
set -ex

echo "Running pre build script..."
echo FFMPEG_ROOT=${FFMPEG_ROOT}

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export BUILD_TYPE="wheel"
setup_env
setup_wheel_python
pip_install numpy future cmake ninja
setup_pip_pytorch_version