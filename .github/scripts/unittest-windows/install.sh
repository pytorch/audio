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

printf "* Installing torch import-time dependencies\n"
pip install numpy

# 1. Install PyTorch
if [ -z "${CUDA_VERSION:-}" ] ; then
    wheel="cpu"
else
    wheel="cu$(python -c "print(''.join(\"${CUDA_VERSION}\".split('.')[:2]))")"
fi
printf "Installing PyTorch\n"
pip install --pre torch --index-url https://download.pytorch.org/whl/${UPLOAD_CHANNEL}/${wheel}

python -c "import torch; print(torch.__version__)"

torch_cuda=$(python -c "import torch; print(torch.cuda.is_available())")
echo torch.cuda.is_available is $torch_cuda

if [ ! -z "${CUDA_VERSION:-}" ] ; then
    if [ "$torch_cuda" == "False" ]; then
        echo "torch with cuda installed but torch.cuda.is_available() is False"
        exit 1
    fi
fi

printf "Installing TorchCodec\n"
# torchcodec nightly has no Windows+CUDA wheels, so we'll use CPU-only
# torchcodec also under CUDA-enabled torch:
pip install --pre torchcodec --index-url https://download.pytorch.org/whl/${UPLOAD_CHANNEL}/cpu
python -c "import torchcodec; print(torchcodec.__version__)"

# 2. Install torchaudio
printf "* Installing fsspec\n"   # TODO: is this required for torchaudio??
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org fsspec

printf "* Installing torchaudio\n"
"$root_dir/packaging/vc_env_helper.bat" pip install . -v --no-build-isolation

# 3. Install Test tools
printf "* Installing test tools\n"
SENTENCEPIECE_DEPENDENCY="sentencepiece"
(
    conda install -y -c conda-forge parameterized 'requests>=2.20'
    # Need to disable shell check since this'll fail out if SENTENCEPIECE_DEPENDENCY is empty
    # shellcheck disable=SC2086
    pip install \
        ${SENTENCEPIECE_DEPENDENCY} \
        Pillow \
        SoundFile \
        coverage \
        expecttest \
        inflect \
        pytest \
        pytest-cov \
        scipy
)
