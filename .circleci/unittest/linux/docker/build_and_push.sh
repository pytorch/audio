#!/usr/bin/env bash

set -euo pipefail

if [ $# -ne 1 ]; then
    printf "Usage %s <CUDA_VERSION>\n\n" "$0"
    exit 1
fi

datestr="$(date "+%Y%m%d")"
if [ "$1" = "cpu" ]; then
    base_image="ubuntu:18.04"
    image="pytorch/torchaudio_unittest_base:manylinux-${datestr}"
else
    base_image="nvidia/cuda:$1-devel-ubuntu18.04"
    docker pull "${base_image}"
    image="pytorch/torchaudio_unittest_base:manylinux-cuda$1-${datestr}"
fi

cd "$( dirname "${BASH_SOURCE[0]}" )"

# docker build also accepts reading from STDIN
# but in that case, no context (other files) can be passed, so we write out Dockerfile
sed "s|BASE_IMAGE|${base_image}|g" Dockerfile > Dockerfile.tmp
docker build -t "${image}" -f Dockerfile.tmp .
docker push "${image}"
