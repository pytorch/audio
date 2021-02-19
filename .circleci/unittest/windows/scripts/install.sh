#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e

root_dir="$(git rev-parse --show-toplevel)"
conda_dir="${root_dir}/conda"
env_dir="${root_dir}/env"

cd "${root_dir}"

# 0. Activate conda env
eval "$("${conda_dir}/Scripts/conda.exe" 'shell.bash' 'hook')"
conda activate "${env_dir}"

# 1. Install PyTorch
if [ -z "${CUDA_VERSION:-}" ] ; then
    cudatoolkit="cpuonly"
else
    version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"
    cudatoolkit="cudatoolkit=${version}"
fi
printf "Installing PyTorch with %s\n" "${cudatoolkit}"
conda install ${CONDA_CHANNEL_FLAGS:-} -y -c "pytorch-${UPLOAD_CHANNEL}" pytorch "${cudatoolkit}"

# TODO: Remove this after packages become available
# Currently there's no librosa package available for Python 3.9, so lets just skip the dependency for now
if [[ $(python --version) != *3.9* ]]; then
    pip install 'librosa>=0.8.0'
fi

# 2. Install torchaudio
printf "* Installing torchaudio\n"
python setup.py install

# 3. Install Test tools
printf "* Installing test tools\n"
conda install -y -c conda-forge codecov pytest pytest-cov
pip install kaldi-io 'librosa>=0.8.0' parameterized PySoundFile scipy
