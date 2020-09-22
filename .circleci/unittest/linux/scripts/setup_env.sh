#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchaudio here, otherwise they also get cached.

set -e

# shellcheck source=../../../../tools/conda_envs/utils.sh
. "$(git rev-parse --show-toplevel)/tools/conda_envs/utils.sh"

install_conda
init_conda

create_env master "${PYTHON_VERSION}"
activate_env master "${PYTHON_VERSION}"
install_build_dependencies

# Buld codecs
mkdir -p third_party/build
(
    cd third_party/build
    cmake ..
    cmake --build .
)
