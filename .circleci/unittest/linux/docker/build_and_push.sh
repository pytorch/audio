#!/usr/bin/env bash

set -euo pipefail

cd "$( dirname "${BASH_SOURCE[0]}" )"

root_dir="$(git rev-parse --show-toplevel)"
cp "${root_dir}"/packaging/build_from_source.sh ./scripts/build_third_parties.sh

tag="manylinux"
image="pytorch/torchaudio_unittest_base:${tag}"
docker build -t "${image}" .
docker push "${image}"
