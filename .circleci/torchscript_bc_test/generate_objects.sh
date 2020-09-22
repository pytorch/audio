#!/usr/bin/env bash

set -e

# shellcheck source=../../tools/conda_envs/utils.sh
. "$(git rev-parse --show-toplevel)/tools/conda_envs/utils.sh"

init_conda

# Move to test directory so that the checked out torchaudio source
# will not shadow the conda-installed version of torchaudio
cd test

for torchaudio in "${TORCHAUDIO_VERSIONS[@]}" ; do
    for python in "${PYTHON_VERSIONS[@]}" ; do
        activate_env "${torchaudio}" "${python}"
        python -m torch.utils.collect_env
        printf "***********************************************************\n"
        printf "* Generating\n"
        printf "  Objects: Python: %s, torchaudio: %s\n" "${python}" "${torchaudio}"
        printf "***********************************************************\n"
        ./torchscript_bc_test/main.py --mode generate --version "${torchaudio}"
    done
done
