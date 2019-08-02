#!/bin/bash

set -ex

if [[ ":$PATH:" == *"conda"* ]]; then
  echo "existing anaconda install in PATH, remove it and run script"
  exit 1
fi

# download and activate anaconda
rm -rf ~/minconda_wheel_env_tmp
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh && \
    chmod +x Miniconda3-latest-MacOSX-x86_64.sh && \
    ./Miniconda3-latest-MacOSX-x86_64.sh -b -p ~/minconda_wheel_env_tmp && \
    rm Miniconda3-latest-MacOSX-x86_64.sh

. ~/minconda_wheel_env_tmp/bin/activate

export TORCHAUDIO_BUILD_VERSION="0.2.0"
export TORCHAUDIO_BUILD_NUMBER="1"
export OUT_DIR=~/torchaudio_wheels

export MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++

# TODO remove when pytorch is good https://github.com/pytorch/pytorch/issues/20030
brew install libomp
CURR_PATH=$(pwd)

cd /tmp
rm -rf audio
git clone https://github.com/pytorch/audio -b v${TORCHAUDIO_BUILD_VERSION}
mkdir audio/third_party

export PREFIX="/tmp"
. $CURR_PATH/build_from_source.sh

cd /tmp/audio

desired_pythons=( "2.7" "3.5" "3.6" "3.7" )
# for each python
for desired_python in "${desired_pythons[@]}"
do
    # create and activate python env
    env_name="env$desired_python"
    conda create -yn $env_name python="$desired_python"
    conda activate $env_name

    pip install torch numpy future

    IS_WHEEL=1 python setup.py clean
    IS_WHEEL=1 python setup.py bdist_wheel
    mkdir -p $OUT_DIR
    cp dist/*.whl $OUT_DIR/
done
