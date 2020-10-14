#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env
python --version

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
conda install -y -c "pytorch-${UPLOAD_CHANNEL}" pytorch ${cudatoolkit}

printf "* Installing torchaudio\n"
BUILD_SOX=1 python setup.py install
