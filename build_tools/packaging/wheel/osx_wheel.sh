#!/bin/bash

set -ex

if [[ ":$PATH:" == *"conda"* ]]; then
    echo "existing anaconda install in PATH, remove it and run script"
    exit 1
fi
# download and activate anaconda
rm -rf ~/minconda_wheel_env_tmp
curl -o Miniconda3-latest-MacOSX-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh && \
    chmod +x Miniconda3-latest-MacOSX-x86_64.sh && \
    ./Miniconda3-latest-MacOSX-x86_64.sh -b -p ~/minconda_wheel_env_tmp && \
    rm Miniconda3-latest-MacOSX-x86_64.sh

. ~/minconda_wheel_env_tmp/bin/activate

if [[ -z "$TORCHAUDIO_BUILD_VERSION" ]]; then
  export TORCHAUDIO_BUILD_VERSION="0.4.0.dev$(date "+%Y%m%d")"
fi
if [[ -z "$TORCHAUDIO_BUILD_NUMBER" ]]; then
  export TORCHAUDIO_BUILD_NUMBER="1"
fi
export OUT_DIR=~/torchaudio_wheels

export MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [[ "$TARGET_COMMIT" == HEAD ]]; then
  # Assume that this script was called from a valid checkout
  WORKDIR="$script_dir/../../.."
else
  WORKDIR="/tmp/audio"
  cd /tmp
  rm -rf audio
  git clone https://github.com/pytorch/audio
  cd audio
  git checkout "$TARGET_COMMIT"
  git submodule update --init --recursive
fi

mkdir "$WORKDIR/third_party"

export PREFIX="$WORKDIR"
. "$script_dir/build_from_source.sh"

cd "$WORKDIR"

ORIG_TORCHAUDIO_PYTORCH_DEPENDENCY_VERSION="$TORCHAUDIO_PYTORCH_DEPENDENCY_VERSION"

desired_pythons=( "2.7" "3.5" "3.6" "3.7" )
# for each python
for desired_python in "${desired_pythons[@]}"
do
    # create and activate python env
    env_name="env$desired_python"
    conda create -yn $env_name python="$desired_python"
    conda activate $env_name

    if [[ -z "$ORIG_TORCHAUDIO_PYTORCH_DEPENDENCY_VERSION" ]]; then
      pip install torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
      # NB: OS X builds don't have local package qualifiers
      # NB: Don't use \+ here, it's not portable
      export TORCHAUDIO_PYTORCH_DEPENDENCY_VERSION="$(pip show torch | grep ^Version: | sed 's/Version:  *//')"
    else
      # NB: We include the nightly channel to, since sometimes we stage
      # prereleases in it.  Those releases should get moved to stable
      # when they're ready
      pip install "torch==$TORCHAUDIO_PYTORCH_DEPENDENCY_VERSION" \
        -f https://download.pytorch.org/whl/torch_stable.html \
        -f https://download.pytorch.org/whl/nightly/torch_nightly.html
    fi
    echo "Building against ${TORCHAUDIO_PYTORCH_DEPENDENCY_VERSION}"

    pip install numpy future
    IS_WHEEL=1 python setup.py clean
    IS_WHEEL=1 python setup.py bdist_wheel
    mkdir -p $OUT_DIR
    cp dist/*.whl $OUT_DIR/
done
