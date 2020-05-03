#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e

if [[ "$OSTYPE" == "msys" ]]; then
    eval "$(./conda/Scripts/conda.exe 'shell.bash' 'hook')"
else
    eval "$(./conda/bin/conda shell.bash hook)"
fi
conda activate ./env

printf "* Installing PyTorch nightly build"
conda install -y -c pytorch-nightly -c conda-forge -c defaults pytorch cpuonly

printf "* Installing torchaudio\n"
if [[ "$(uname)" == "Linux" ]]; then
    # Link codecs present at /third_party. See Dockerfile for how this is built
    ln -fs /third_party ./third_party
fi
IS_CONDA=true python setup.py develop
