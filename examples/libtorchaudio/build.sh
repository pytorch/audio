#!/usr/bin/env bash

set -eux

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
build_dir="${this_dir}/build"

mkdir -p "${build_dir}"
cd "${build_dir}"

git submodule update
cmake -GNinja \
      -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')" \
      ..
cmake --build .
