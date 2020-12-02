#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

if [ -z "${CUDA_VERSION:-}" ] ; then
    case "$(uname -s)" in
        Darwin*) cudatoolkit="";;
        *) cudatoolkit="cpuonly"
    esac
else
    version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"
    cudatoolkit="cudatoolkit=${version}"
fi
printf "Installing PyTorch with %s\n" "${cudatoolkit}"
conda install ${CONDA_CHANNEL_FLAGS:-} -y -c "pytorch-${UPLOAD_CHANNEL}" pytorch ${cudatoolkit}

printf "* Installing dependencies for test\n"
CONDA_PKGS="librosa>=0.8.0"
PIP_PKGS=""
if [[ $(python --version) = *3.9* ]]; then
    CONDA_PKGS=""
    # librosa doesn't have conda packages for python 3.9+
    PIP_PKGS="librosa>=0.8.0"
fi
conda install -y -c conda-forge pytest pytest-cov codecov scipy parameterized ${CONDA_PKGS}
pip install kaldi-io ${PIP_PKGS}

printf "* Building codecs\n"
mkdir -p third_party/build
(
    cd third_party/build
    cmake ..
    cmake --build .
)

printf "* Installing torchaudio\n"
BUILD_SOX=1 python setup.py install
