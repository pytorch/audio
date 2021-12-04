#!/usr/bin/env bash

set -ex

# shellcheck disable=SC1091
source ./packaging/pkg_helpers.bash
export NO_CUDA_PACKAGE=1
setup_env 0.8.0
setup_wheel_python
# Starting 0.10, `pip install pytorch` defaults to ROCm.
export PYTORCH_VERSION_SUFFIX="+cpu"
setup_pip_pytorch_version
# pytorch is already installed
pip install --no-deps ~/workspace/torchaudio*
