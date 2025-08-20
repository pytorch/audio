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
"$root_dir/packaging/vc_env_helper.bat" pip install . -v --no-build-isolation

# 3. Install Test tools
printf "* Installing test tools\n"
pip install parameterized requests coverage pytest pytest-cov scipy numpy expecttest
