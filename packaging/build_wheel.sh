#!/bin/bash
set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export BUILD_TYPE="wheel"
export NO_CUDA_PACKAGE=1
setup_env 0.6.0
setup_wheel_python
if [[ "$OSTYPE" == "msys" ]]; then
    "$script_dir/download_deps.sh" "$(pwd)"  # Download static dependencies
else
    "$script_dir/build_from_source.sh" "$(pwd)"  # Build static dependencies
fi
pip_install numpy future
setup_pip_pytorch_version
python setup.py clean
if [[ "$OSTYPE" == "msys" ]]; then
    python_tag="$(echo "cp$PYTHON_VERSION" | tr -d '.')"
    IS_WHEEL=1 python setup.py bdist_wheel --plat-name win_amd64 --python-tag $python_tag
else
    IS_WHEEL=1 python setup.py bdist_wheel
fi
