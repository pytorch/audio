#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchaudio here, otherwise they also get cached.

set -ex

root_dir="$(git rev-parse --show-toplevel)"
conda_dir="${root_dir}/conda"
env_dir="${root_dir}/env"

cd "${root_dir}"

case "$(uname -s)" in
    Darwin*) os=MacOSX;;
    *) os=Linux
esac

# 1. Install conda at ./conda
if [ ! -d "${conda_dir}" ]; then
    printf "* Installing conda\n"
    wget --quiet -O miniconda.sh "http://repo.continuum.io/miniconda/Miniconda3-latest-${os}-x86_64.sh"
    bash ./miniconda.sh -b -f -p "${conda_dir}"
    eval "$("${conda_dir}/bin/conda" shell.bash hook)"
    conda update --quiet -y conda
    printf "* Updating the base Python version to %s\n" "${PYTHON_VERSION}"
    conda install --quiet -y python="${PYTHON_VERSION}"
else
    eval "$("${conda_dir}/bin/conda" shell.bash hook)"
fi


# 2. Create test environment at ./env
if [ ! -d "${env_dir}" ]; then
    printf "* Creating a test environment with PYTHON_VERSION=%s\n" "${PYTHON_VERSION}\n"
    conda create --prefix "${env_dir}" -y python="${PYTHON_VERSION}"
fi
conda activate "${env_dir}"
NUMPY_PIN="1.11"
if [[ "${PYTHON_VERSION}" = "3.9" ]]; then
    NUMPY_PIN="1.20"
fi
printf "* Installing numpy>=%s\n" "${NUMPY_PIN}\n"
conda install -y -c conda-forge "numpy>=${NUMPY_PIN}"

# 3. Install minimal build tools
pip --quiet install cmake ninja
