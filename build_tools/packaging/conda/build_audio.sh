#!/usr/bin/env bash
if [[ -x "/remote/anaconda_token" ]]; then
    . /remote/anaconda_token || true
fi

set -ex

 # Function to retry functions that sometimes timeout or have flaky failures
retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

export TORCHAUDIO_BUILD_VERSION="0.2.0"
export TORCHAUDIO_BUILD_NUMBER=1

SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

if [[ -z "$WIN_PACKAGE_WORK_DIR" ]]; then
    WIN_PACKAGE_WORK_DIR="$(echo $(pwd -W) | tr '/' '\\')\\tmp_conda_$(date +%H%M%S)"
fi

if [[ "$OSTYPE" == "msys" ]]; then
    mkdir -p "$WIN_PACKAGE_WORK_DIR" || true
    audio_rootdir="$(realpath ${WIN_PACKAGE_WORK_DIR})/torchaudio-src"
    git config --system core.longpaths true
else
    audio_rootdir="$(pwd)/torchaudio-src"
fi

if [[ ! -d "$audio_rootdir" ]]; then
    rm -rf "$audio_rootdir"
    git clone "https://github.com/pytorch/audio" "$audio_rootdir"
    pushd "$audio_rootdir"
    git checkout v$TORCHAUDIO_BUILD_VERSION
    popd
fi

 cd "$SOURCE_DIR"

 if [[ "$OSTYPE" == "msys" ]]; then
    export tmp_conda="${WIN_PACKAGE_WORK_DIR}\\conda"
    export miniconda_exe="${WIN_PACKAGE_WORK_DIR}\\miniconda.exe"
    rm -rf "$tmp_conda"
    rm -f "$miniconda_exe"
    curl -sSk https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe -o "$miniconda_exe"
    "$SOURCE_DIR/install_conda.bat" && rm "$miniconda_exe"
    pushd $tmp_conda
    export PATH="$(pwd):$(pwd)/Library/usr/bin:$(pwd)/Library/bin:$(pwd)/Scripts:$(pwd)/bin:$PATH"
    popd
    # We have to skip 3.17 because of the following bug.
    # https://github.com/conda/conda-build/issues/3285
    retry conda install -yq conda-build
fi

ANACONDA_USER=pytorch
conda config --set anaconda_upload no

# "$desired_cuda" == 'cpu'
export TORCHAUDIO_PACKAGE_SUFFIX=""
export CONDA_CUDATOOLKIT_CONSTRAINT=""
export CUDA_VERSION="None"
if [[ "$OSTYPE" != "darwin"* ]]; then
    export TORCHAUDIO_PACKAGE_SUFFIX="-cpu"
else
  export MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++
fi

if [[ "$OSTYPE" == "msys" ]]; then
    time conda build -c $ANACONDA_USER --no-anaconda-upload vs2017
else
    time conda build -c $ANACONDA_USER --no-anaconda-upload --python 2.7 torchaudio
fi

time conda build -c $ANACONDA_USER --no-anaconda-upload --python 3.5 torchaudio
time conda build -c $ANACONDA_USER --no-anaconda-upload --python 3.6 torchaudio
time conda build -c $ANACONDA_USER --no-anaconda-upload --python 3.7 torchaudio

 set +e
