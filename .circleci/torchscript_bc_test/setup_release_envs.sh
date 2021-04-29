#!/usr/bin/env bash

set -e

cd "$( dirname "${BASH_SOURCE[0]}" )"
. "common.sh"

install_conda
init_conda

# Install torchaudio environments
for torchaudio in "${TORCHAUDIO_VERSIONS[@]}" ; do
    for python in "${PYTHON_VERSIONS[@]}" ; do
        create_env "${torchaudio}" "${python}"
        activate_env "${torchaudio}" "${python}"
        install_release "${torchaudio}"
    done
done
