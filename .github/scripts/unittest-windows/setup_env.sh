#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchaudio here, otherwise they also get cached.

set -euxo pipefail

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
root_dir="$(git rev-parse --show-toplevel)"
conda_dir="${root_dir}/conda_"
env_dir="${root_dir}/env"

cd "${root_dir}"

# 1. Install conda at ./conda
if [ ! -d "${conda_dir}" ]; then
    printf "* Installing conda\n"
    export tmp_conda="$(echo $conda_dir | tr '/' '\\')"
    echo "tmp_conda=$tmp_conda"
    export miniconda_exe="$(echo $root_dir | tr '/' '\\')\\miniconda.exe"
    curl --silent --output miniconda.exe https://repo.anaconda.com/miniconda/Miniconda3-py39_25.7.0-2-Windows-x86_64.exe -O
    "$this_dir/install_conda.bat"
    echo "2: Content of tmp_conda: $(ls ${tmp_conda})"
    unset tmp_conda
    unset miniconda_exe
fi

echo "2: Content of conda_dir: $(ls ${conda_dir})"
echo "2: Content of conda_dir/Lib: $(ls ${conda_dir}/Lib)"

eval "$("${conda_dir}/_conda.exe" 'shell.bash' 'hook')"

# 2. Create test environment at ./env
if [ ! -d "${env_dir}" ]; then
    printf "* Creating a test environment with PYTHON_VERSION=%s\n" "${PYTHON_VERSION}"
    conda create --prefix "${env_dir}" -y python="${PYTHON_VERSION}"
fi
conda activate "${env_dir}"

# 3. Install minimal build tools
pip --quiet install cmake ninja
# conda install --quiet -y 'ffmpeg>=4.1'
