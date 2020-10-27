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

case "$(uname -s)" in
    Darwin*) os=MacOSX;;
    *) os=Linux
esac

# 0. Activate conda env
eval "$("${conda_dir}/bin/conda" shell.bash hook)"
conda activate "${env_dir}"

# 1. Install PyTorch
if [ -z "${CUDA_VERSION:-}" ] ; then
    if [ "${os}" == MacOSX ] ; then
        cudatoolkit=''
    else
        cudatoolkit="cpuonly"
    fi
else
    version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"
    cudatoolkit="cudatoolkit=${version}"
fi
printf "Installing PyTorch with %s\n" "${cudatoolkit}"
conda install -y -c "pytorch-${UPLOAD_CHANNEL}" pytorch ${cudatoolkit}

# 2. Install torchaudio
printf "* Installing torchaudio\n"
BUILD_SOX=1 python setup.py install

# 3. Install Test tools
printf "* Installing test tools\n"
if [ "${os}" == Linux ] ; then
    # TODO: move this to docker
    apt install -y -q libsndfile1
    conda install -y -c conda-forge codecov pytest pytest-cov
    pip install kaldi-io 'librosa>=0.8.0' parameterized SoundFile scipy
else
    pip install kaldi-io SoundFile
    conda install -y -c conda-forge codecov pytest pytest-cov 'librosa>=0.8.0' parameterized scipy
fi
