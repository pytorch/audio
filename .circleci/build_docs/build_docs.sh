#!/usr/bin/env bash

set -ex
# shellcheck disable=SC1091


source ./packaging/pkg_helpers.bash
export NO_CUDA_PACKAGE=1
setup_env 0.8.0
setup_wheel_python

pushd docs
VERSION=$1
pip install -r requirements.txt
make html 
popd
