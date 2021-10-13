#!/usr/bin/env bash
set -ex

torch_cuda_version=$(python -c "import torch; print(torch.version.cuda)")
echo torch.cuda.is_available is $torch_cuda_version

shopt -s nocasematch
echo CUDA_VERSION is "$CUDA_VERSION"
if [ ! -z "${CUDA_VERSION:-}" ] ; then
    if [ "$torch_cuda_version" == "None" ]; then
        echo "We wan't build torch auido with cuda but the installed pytorch isn't with cuda"
        exit 1
    fi
fi
shopt -u nocasematch

python setup.py install --single-version-externally-managed --record=record.txt
