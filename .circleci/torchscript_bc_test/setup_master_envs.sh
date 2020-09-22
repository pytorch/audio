#!/usr/bin/env bash

set -e

# shellcheck source=../../tools/conda_envs/utils.sh
. "$(git rev-parse --show-toplevel)/tools/conda_envs/utils.sh"

install_conda
init_conda

# Install torchaudio environments
for python in "${PYTHON_VERSIONS[@]}" ; do
    create_env master "${python}"
    activate_env master "${python}"
    install_build_dependencies
    build_master
done
