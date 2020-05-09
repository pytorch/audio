#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e

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
/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2019/Community/VC/Auxiliary/Build/vcvarsall.bat x64
python setup.py develop
