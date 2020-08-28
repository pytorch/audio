#!/usr/bin/env bash

set -e

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
build_dir="${this_dir}/tmp"

# rm -rf "${build_dir}"
mkdir -p "${build_dir}"

(
cd "${build_dir}"
cmake -GNinja \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
     -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')" \
    _GLIBCXX_USE_CXX11_ABI=1 \
    -DBUILD_PYTHON_EXTENSION:BOOL=ON \
    -DBUILD_LIBTORCHAUDIO:BOOL=OFF \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY:PATH="${this_dir}/torchaudio" \
    -DCMAKE_INSTALL_PREFIX:PATH="${this_dir}/torchaudio" \
    ..

cmake --build . --target _torchaudio
)

# BUILD_SOX=1 python setup.py clean develop
# cd "${this_dir}"
# rm -rf torchaudio/_torchaudio.so
# cp tmp/install/lib/_torchaudio.so torchaudio/
python -c 'import torchaudio'
