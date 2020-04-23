#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used, and we do not want to fixiate on the version

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

printf "* Installing PyTorch Nightly"
conda install -c pytorch-nightly pytorch cpuonly

printf "* Setting up torchaudio\n"
IS_CONDA=true python setup.py develop
