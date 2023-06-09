#!/usr/bin/env bash
set -euxo pipefail

if [ -z "${CUDA_VERSION:-}" ] ; then
    version="cpu"
else
    version="$CUDA_VERSION"
fi

# Don't use if [[ "$version" == "cpu" ]]; then exit 0 fi.
# It would exit the shell. One result is cpu tests would not run if the shell exit.
# Unless there's an error, Don't exit.
if [[ "$version" != "cpu" ]]; then
    # set cuda envs
    export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${version}/bin:/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${version}/libnvvp:$PATH"
    export CUDA_PATH_V${version/./_}="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v${version}"
    export CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v${version}"

    if  [ ! -d "$CUDA_PATH" ]
    then
        echo "$CUDA_PATH" does not exist
        exit 1
    fi

    # check cuda driver version
    for path in '/c/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe' /c/Windows/System32/nvidia-smi.exe; do
        if [[ -x "$path" ]]; then
            "$path" || echo "true";
            break
        fi
    done

    which nvcc
    nvcc --version
    env | grep CUDA
fi
