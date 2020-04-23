#!/usr/bin/env bash

# This script is for setting up environment for running unit test on CircleCI.
# To speed up the CI time, the result of environment is cached.
# PyTorch is not included here, so that it won't be cached.

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
conda activate "${env_dir}"
conda env update --file "${this_dir}/environment.yml" --prune

# 3. Build codecs at ./third_party
if [ ! -d "./third_party" ]; then
    printf "* Building Codecs"
    ./packaging/build_from_source.sh "$PWD"
fi
