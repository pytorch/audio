#!/usr/bin/env bash

set -e

cd "$( dirname "${BASH_SOURCE[0]}" )"
. "common.sh"

install_conda
init_conda

# Install torchaudio environments
for python in "${PYTHON_VERSIONS[@]}" ; do
    create_env master "${python}"
    activate_env master "${python}"
    install_build_dependencies "${python}"
    build_master "${python}"
done
