#!/usr/bin/env bash

set -euo pipefail

if [ $# != 1 ]; then
    printf "Usage %s <cu_version>\n\n" "$0"
    printf "Build a torchaudio Docker image for unittest and push it to Docker hub.\n\n"
    printf "Expected values for <cu_version> are '92', '100', '101' or '102'\n"
    printf "This value controll which base image will be used. (i.e. pytorch/manylinux-cudaXXX"
    exit 1;
fi

cu_version="$1"

cd "$( dirname "${BASH_SOURCE[0]}" )"

root_dir="$(git rev-parse --show-toplevel)"
cp "${root_dir}"/packaging/build_from_source.sh ./scripts/build_third_parties.sh

tag="manylinux-cuda${cu_version}"
base_image="pytorch/${tag}"
image="pytorch/torchaudio_unittest_base:${tag}"
# docker build also accepts reading from STDIN
# but in that case, no context (other files) can be passed, so we write out Dockerfile
sed "s|BASE_IMAGE|${base_image}|g" Dockerfile_template > Dockerfile
docker build -t "${image}" .
docker push "${image}"
