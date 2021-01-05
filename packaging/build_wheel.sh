#!/bin/bash
set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export BUILD_TYPE="wheel"
export NO_CUDA_PACKAGE=1
setup_env 0.8.0
setup_wheel_python
pip_install numpy future
setup_pip_pytorch_version
python setup.py clean
if [[ "$OSTYPE" == "msys" ]]; then
    python_tag="$(echo "cp$PYTHON_VERSION" | tr -d '.')"
    python setup.py bdist_wheel --plat-name win_amd64 --python-tag $python_tag
else
    BUILD_TRANSDUCER=1 BUILD_SOX=1 python setup.py bdist_wheel
fi
