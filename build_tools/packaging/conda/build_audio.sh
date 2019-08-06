#!/usr/bin/env bash
if [[ -x "/remote/anaconda_token" ]]; then
    . /remote/anaconda_token || true
fi

set -ex

 # Function to retry functions that sometimes timeout or have flaky failures
retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

if [[ -z "$TORCHAUDIO_BUILD_VERSION" ]]; then
  export TORCHAUDIO_BUILD_VERSION="0.4.0.dev$(date "+%Y%m%d")"
fi
if [[ -z "$TORCHAUDIO_BUILD_NUMBER" ]]; then
  export TORCHAUDIO_BUILD_NUMBER=1
fi

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

. "$script_dir/../setup_workdir"
export TORCHAUDIO_GITHUB_ROOT_DIR="$WORKDIR"

cd "$script_dir"

ANACONDA_USER=pytorch-nightly
conda config --set anaconda_upload no
conda config --set channel_priority strict

LATEST_PYTORCH_NIGHTLY_VERSION=$(conda search --json 'pytorch[channel=pytorch-nightly]' | python "$script_dir/get-latest.py")
export CONDA_PYTORCH_CONSTRAINT="    - pytorch ==${LATEST_PYTORCH_NIGHTLY_VERSION}"
export CONDA_CUDATOOLKIT_CONSTRAINT=""
export CUDA_VERSION="None"
if [[ "$OSTYPE" == "darwin"* ]]; then
  export MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++
fi

time conda build -c $ANACONDA_USER --no-anaconda-upload --python 2.7 torchaudio
time conda build -c $ANACONDA_USER --no-anaconda-upload --python 3.5 torchaudio
time conda build -c $ANACONDA_USER --no-anaconda-upload --python 3.6 torchaudio
time conda build -c $ANACONDA_USER --no-anaconda-upload --python 3.7 torchaudio
