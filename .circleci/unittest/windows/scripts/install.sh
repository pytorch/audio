#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
root_dir="$(git rev-parse --show-toplevel)"

cd "${root_dir}"

eval "$(./conda/Scripts/conda.exe 'shell.bash' 'hook')"
conda activate ./env

if [ -z "${CUDA_VERSION:-}" ] ; then
    cudatoolkit="cpuonly"
else
    version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"
    cudatoolkit="cudatoolkit=${version}"
fi
printf "Installing PyTorch with %s\n" "${cudatoolkit}"
conda install -y -c pytorch-nightly pytorch "${cudatoolkit}"

printf "* Installing torchaudio\n"
curl --retry 3 https://s3.amazonaws.com/ossci-windows/torchaudio_deps.7z --output /tmp/torchaudio_deps.7z
7z x /tmp/torchaudio_deps.7z -othird_party
IS_CONDA=true "$this_dir/install.bat"
