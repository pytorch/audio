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
# [2021/06/22 Temporary workaround] Disabling the original installation
# The orignal, conda-based instartion is working for GPUs, but not for CPUs
# For CPUs we use pip-based installation
# if [ -z "${CUDA_VERSION:-}" ] ; then
#     if [ "${os}" == MacOSX ] ; then
#         cudatoolkit=''
#     else
#         cudatoolkit="cpuonly"
#     fi
# else
#     version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"
#     cudatoolkit="cudatoolkit=${version}"
# fi
# printf "Installing PyTorch with %s\n" "${cudatoolkit}"
# (
#     set -x
#     conda install ${CONDA_CHANNEL_FLAGS:-} -y -c "pytorch-${UPLOAD_CHANNEL}" "pytorch-${UPLOAD_CHANNEL}::pytorch" ${cudatoolkit}
# )

if [ "${os}" == MacOSX ] || [ -z "${CUDA_VERSION:-}" ] ; then
    device="cpu"
    printf "Installing PyTorch with %s\n" "$device}"
    (
        set -x
        pip install --pre torch==1.10.0.dev20210618 -f "https://download.pytorch.org/whl/nightly/${device}/torch_nightly.html"
    )
else
    version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"
    cudatoolkit="cudatoolkit=${version}"
    printf "Installing PyTorch with %s\n" "${cudatoolkit}"
    (
        set -x
        conda install ${CONDA_CHANNEL_FLAGS:-} -y -c "pytorch-${UPLOAD_CHANNEL}" "pytorch-${UPLOAD_CHANNEL}::pytorch" ${cudatoolkit}
    )
fi

# 2. Install torchaudio
printf "* Installing torchaudio\n"
git submodule update --init --recursive
BUILD_TRANSDUCER=1 BUILD_SOX=1 python setup.py install

# 3. Install Test tools
printf "* Installing test tools\n"
NUMBA_DEV_CHANNEL=""
if [[ "$(python --version)" = *3.9* ]]; then
    # Numba isn't available for Python 3.9 except on the numba dev channel and building from source fails
    # See https://github.com/librosa/librosa/issues/1270#issuecomment-759065048
    NUMBA_DEV_CHANNEL="-c numba/label/dev"
fi
# Note: installing librosa via pip fail because it will try to compile numba.
(
    set -x
    conda install -y -c conda-forge ${NUMBA_DEV_CHANNEL} 'librosa>=0.8.0' parameterized 'requests>=2.20'
    pip install kaldi-io SoundFile coverage pytest pytest-cov scipy transformers
)
# Install fairseq
git clone https://github.com/pytorch/fairseq
cd fairseq
git checkout e6eddd80
pip install .
