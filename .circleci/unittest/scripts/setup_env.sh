#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchaudio here, otherwise they also get cached.

set -e

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
root_dir="$(git rev-parse --show-toplevel)"
conda_dir="${root_dir}/conda"
env_dir="${root_dir}/env"

cd "${root_dir}"

# 1. Install conda at ./conda
if [ ! -d "${conda_dir}" ]; then
    printf "* Installing conda\n"
    wget -O miniconda.sh http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash ./miniconda.sh -b -f -p "${conda_dir}"
fi
printf "* Checking conda update\n"
eval "$(${conda_dir}/bin/conda shell.bash hook)"
conda update -n base -c defaults conda

# 2. Create test environment at ./env
if [ ! -d "${env_dir}" ]; then
    printf "* Creating a test environment\n"
    conda create --prefix "${env_dir}" -y python="$PYTHON_VERSION"
fi
printf "* Installing dependencies (except PyTorch)\n"
conda activate "${env_dir}"
conda env update --file "${this_dir}/environment.yml" --prune

# 3. Link codecs present at /third_party
# See Dockerfile for how this is built
ln -fs /third_party ./third_party
