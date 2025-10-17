#!/usr/bin/env bash
set -ex

torch_cuda_version=$(python -c "import torch; print(torch.version.cuda)")
echo "torch.cuda.version is $torch_cuda_version"

echo USE_CUDA is "$USE_CUDA"

shopt -s nocasematch
if [ "${USE_CUDA}" == "1" ] ; then
    if [ "$torch_cuda_version" == "None" ]; then
        echo "We want to build torch auido with cuda but the installed pytorch isn't with cuda"
        exit 1
    fi
fi
shopt -u nocasematch
python -m pip install . -vv --no-build-isolation
