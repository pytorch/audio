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
    export tmp_conda="$(echo $conda_dir | tr '/' '\\')"
    export miniconda_exe="$(echo $root_dir | tr '/' '\\')\\miniconda.exe"
    curl --silent --output miniconda.exe https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -O
    "$this_dir/install_conda.bat"
    unset tmp_conda
    unset miniconda_exe
    eval "$("${conda_dir}/Scripts/conda.exe" 'shell.bash' 'hook')"
    conda update --quiet -y conda
    printf "* Updating the base Python version to %s\n" "${PYTHON_VERSION}"
    conda install --quiet -y python="$PYTHON_VERSION"
else
    eval "$("${conda_dir}/Scripts/conda.exe" 'shell.bash' 'hook')"
fi

# 2. Create test environment at ./env
if [ ! -d "${env_dir}" ]; then
    printf "* Creating a test environment with PYTHON_VERSION=%s\n" "${PYTHON_VERSION}"
    conda create --prefix "${env_dir}" -y python="${PYTHON_VERSION}"
fi
conda activate "${env_dir}"
NUMPY_PIN="1.11"
if [[ "${PYTHON_VERSION}" = "3.9" ]]; then
    NUMPY_PIN="1.20"
fi
printf "* Installing numpy>=%s\n" "${NUMPY_PIN}\n"
conda install -y -c conda-forge numpy>="${NUMPY_PIN}"
