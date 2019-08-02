#!/usr/bin/env bash
if [[ -x "/remote/anaconda_token" ]]; then
    . /remote/anaconda_token || true
fi

set -ex

 # Function to retry functions that sometimes timeout or have flaky failures
retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

export TORCHAUDIO_BUILD_VERSION="0.4.0.dev$(date "+%Y%m%d")"
export TORCHAUDIO_BUILD_NUMBER=1

SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

audio_rootdir="$(pwd)/torchaudio-src"

if [[ ! -d "$audio_rootdir" ]]; then
    rm -rf "$audio_rootdir"
    git clone "https://github.com/pytorch/audio" "$audio_rootdir" -b pr/conda-nightly
fi

export TORCHAUDIO_GITHUB_ROOT_DIR="$audio_rootdir"

cd "$SOURCE_DIR"

ANACONDA_USER=pytorch
conda config --set anaconda_upload no

# TODO: unhardcode
export CONDA_PYTORCH_CONSTRAINT="    - pytorch-nightly ==1.2.0.dev20190802+cpu"
export CONDA_CUDATOOLKIT_CONSTRAINT=""
export CUDA_VERSION="None"
if [[ "$OSTYPE" == "darwin"* ]]; then
  export MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++
fi

time conda build -c $ANACONDA_USER --no-anaconda-upload --python 2.7 torchaudio
# Currently disabled, see https://github.com/pytorch/pytorch/issues/23738
# time conda build -c $ANACONDA_USER --no-anaconda-upload --python 3.5 torchaudio
time conda build -c $ANACONDA_USER --no-anaconda-upload --python 3.6 torchaudio
time conda build -c $ANACONDA_USER --no-anaconda-upload --python 3.7 torchaudio

 set +e
