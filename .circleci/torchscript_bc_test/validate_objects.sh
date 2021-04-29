#!/usr/bin/env bash

set -e

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
pushd "${this_dir}"
. "common.sh"
popd

init_conda

# Move to test directory so that the checked out torchaudio source
# will not shadow the conda-installed version of torchaudio
cd test

# Validate torchscript objects for each
for runtime_python in "${PYTHON_VERSIONS[@]}" ; do
    activate_env master "${runtime_python}"
    python -m torch.utils.collect_env
    for object_torchaudio in "${TORCHAUDIO_VERSIONS[@]}" ; do
        for object_python in "${PYTHON_VERSIONS[@]}" ; do
            printf "***********************************************************\n"
            printf "* Validating\n"
            printf "  Runtime: Python: %s, torchaudio: master (%s)\n" "${runtime_python}" "$(python -c 'import torchaudio;print(torchaudio.__version__)')"
            printf "  Objects: Python: %s, torchaudio: %s\n" "${object_python}" "${object_torchaudio}"
            printf "***********************************************************\n"
            ./torchscript_bc_test/main.py --mode validate --version "${object_torchaudio}"
        done
    done
done
