#!/usr/bin/env bash

set -euo pipefail

if [ $# -ne 1 ]; then
    printf "Usage %s <CUDA_VERSION>\n\n" "$0"
    exit 1
fi

if [ "$1" = "cpu" ]; then
    base_image="ubuntu:18.04"
    image="pytorch/torchaudio_unittest_base:manylinux"
elif [[ "$1" =~ ^(9.2|10.1)$ ]]; then
    base_image="nvidia/cuda:$1-runtime-ubuntu18.04"
    image="pytorch/torchaudio_unittest_base:manylinux-cuda$1"
else
    printf "Unexpected <CUDA_VERSION> string: %s" "$1"
    exit 1;
fi

cd "$( dirname "${BASH_SOURCE[0]}" )"

root_dir="$(git rev-parse --show-toplevel)"
cp "${root_dir}"/packaging/build_from_source.sh ./scripts/build_third_parties.sh

# docker build also accepts reading from STDIN
# but in that case, no context (other files) can be passed, so we write out Dockerfile
sed "s|BASE_IMAGE|${base_image}|g" Dockerfile > Dockerfile.tmp
docker build -t "${image}" -f Dockerfile.tmp .
docker push "${image}"
