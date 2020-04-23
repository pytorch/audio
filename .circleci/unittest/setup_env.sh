#!/usr/bin/env bash

# This script is for setting up environment for running unit test on CircleCI.

set -e

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used, and we do not want to fixiate on the version

conda_location="${HOME}/miniconda3"
root_dir="$(git rev-parse --show-toplevel)"
this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd "${root_dir}"

printf "* Installing conda\n"
wget -O miniconda.sh http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./miniconda.sh -b -f -p "${conda_location}"
eval "$(${conda_location}/bin/conda shell.bash hook)"
conda init
conda update -n base -c defaults conda

printf "* Creating a test environment\n"
conda create --prefix ./env -y python="$PYTHON_VERSION"
source activate ./env
conda env update --file "${this_dir}/environment.yml" --prune

printf "* Setting up torchaudio\n"
./packaging/build_from_source.sh "$PWD"
IS_CONDA=true python setup.py develop
