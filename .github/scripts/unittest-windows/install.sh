#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -euxo pipefail

root_dir="$(git rev-parse --show-toplevel)"
conda_dir="${root_dir}/conda"
env_dir="${root_dir}/env"
this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd "${root_dir}"

# 0. Activate conda env
eval "$("${conda_dir}/Scripts/conda.exe" 'shell.bash' 'hook')"
conda activate "${env_dir}"

source "$this_dir/set_cuda_envs.sh"

# 1. Install PyTorch
if [ -z "${CUDA_VERSION:-}" ] ; then
    cudatoolkit="cpuonly"
    version="cpu"
else
    version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"
    cudatoolkit="pytorch-cuda=${version}"
fi
printf "Installing PyTorch with %s\n" "${cudatoolkit}"
conda install -y -c "pytorch-${UPLOAD_CHANNEL}" -c nvidia pytorch "${cudatoolkit}"  pytest pybind11

torch_cuda=$(python -c "import torch; print(torch.cuda.is_available())")
echo torch.cuda.is_available is $torch_cuda

if [ ! -z "${CUDA_VERSION:-}" ] ; then
    if [ "$torch_cuda" == "False" ]; then
        echo "torch with cuda installed but torch.cuda.is_available() is False"
        exit 1
    fi
fi

# 2. Install torchaudio
printf "* Installing fsspec\n"
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org fsspec

printf "* Installing torchaudio\n"
"$root_dir/packaging/vc_env_helper.bat" python setup.py install

# 3. Install Test tools
printf "* Installing test tools\n"
NUMBA_DEV_CHANNEL=""
SENTENCEPIECE_DEPENDENCY="sentencepiece"
case "$(python --version)" in
    *3.9*)
        # Numba isn't available for Python 3.9 except on the numba dev channel and building from source fails
        # See https://github.com/librosa/librosa/issues/1270#issuecomment-759065048
        NUMBA_DEV_CHANNEL="-c numba/label/dev"
        ;;
    *3.10*)
        # Don't install sentencepiece, no python 3.10 dependencies available for windows yet
        SENTENCEPIECE_DEPENDENCY=""
        NUMBA_DEV_CHANNEL="-c numba/label/dev"
        ;;
esac
# Note: installing librosa via pip fail because it will try to compile numba.
(
    conda install -y -c conda-forge ${NUMBA_DEV_CHANNEL} 'librosa==0.10.0' parameterized 'requests>=2.20'
    # Need to disable shell check since this'll fail out if SENTENCEPIECE_DEPENDENCY is empty
    # shellcheck disable=SC2086
    pip install \
        ${SENTENCEPIECE_DEPENDENCY} \
        Pillow \
        SoundFile \
        coverage \
        expecttest \
        inflect \
        kaldi-io \
        pytest \
        pytest-cov \
        pytorch-lightning \
        'scipy==1.7.3' \
        unidecode \
        'protobuf<4.21.0' \
        demucs \
        tinytag \
        pyroomacoustics \
        flashlight-text \
        git+https://github.com/kpu/kenlm/
)
# Install fairseq
git clone https://github.com/pytorch/fairseq
cd fairseq
git checkout e47a4c8
pip install .
