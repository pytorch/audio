#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

printf "* Installing PyTorch nightly build"
conda install -y -c pytorch-nightly pytorch cpuonly

printf "* Installing torchaudio\n"
# Link codecs present at /third_party. See Dockerfile for how this is built
ln -fs /third_party ./third_party
IS_CONDA=true python setup.py develop
