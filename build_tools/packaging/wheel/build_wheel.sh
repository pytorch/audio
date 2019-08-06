#!/bin/bash

# Assume we are in a valid Python environment that we want to build the wheel in

if [[ -z "$OUT_DIR" ]]; then
  export OUT_DIR="/tmp/torchaudio-build"
fi
if [[ -z "$TORCHAUDIO_BUILD_VERSION" ]]; then
  export TORCHAUDIO_BUILD_VERSION="0.4.0.dev$(date "+%Y%m%d")"
fi
if [[ -z "$TORCHAUDIO_BUILD_NUMBER" ]]; then
  export TORCHAUDIO_BUILD_NUMBER="1"
fi
if [[ "$(uname)" == Darwin ]]; then
  export MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++
fi

if [[ -z "$TORCHAUDIO_PYTORCH_DEPENDENCY_VERSION" ]]; then
  is_nightly=1  # to unset later
  pip install --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
  # CPU/CUDA variants of PyTorch have ABI compatible PyTorch.  Therefore, we
  # strip off the local package qualifier.  Also, we choose to build against
  # the CPU build, because it takes less time to download.
  export TORCHAUDIO_PYTORCH_DEPENDENCY_VERSION="$(pip show torch | grep ^Version: | sed 's/Version: \+//' | sed 's/+.\+//')"
else
  is_nightly=
  # NB: We include the nightly channel to, since sometimes we stage
  # prereleases in it.  Those releases should get moved to stable
  # when they're ready
  pip install "torch==$TORCHAUDIO_PYTORCH_DEPENDENCY_VERSION" \
    -f https://download.pytorch.org/whl/torch_stable.html \
    -f https://download.pytorch.org/whl/nightly/torch_nightly.html
fi
echo "Building against ${TORCHAUDIO_PYTORCH_DEPENDENCY_VERSION}"

# NB: do not actually install requirements.txt; that is only needed for
# testing
pip install numpy future
IS_WHEEL=1 python setup.py clean
IS_WHEEL=1 python setup.py bdist_wheel
mkdir -p $OUT_DIR
cp dist/*.whl $OUT_DIR/
