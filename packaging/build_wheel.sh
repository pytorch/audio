#!/bin/bash
set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export BUILD_TYPE="wheel"
export NO_CUDA_PACKAGE=1
setup_env 0.5.0
setup_wheel_python
"$script_dir/build_from_source.sh" "$(pwd)"  # Build static dependencies
pip_install numpy future
setup_pip_pytorch_version
python setup.py clean
IS_WHEEL=1 python setup.py bdist_wheel
