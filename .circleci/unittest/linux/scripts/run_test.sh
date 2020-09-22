#!/usr/bin/env bash

set -e

# shellcheck source=../../../../tools/conda_envs/utils.sh
. "$(git rev-parse --show-toplevel)/tools/conda_envs/utils.sh"

init_conda
activate_env master "${PYTHON_VERSION}"

python -m torch.utils.collect_env
export TORCHAUDIO_TEST_FAIL_IF_NO_EXTENSION=1
export PATH="${PWD}/third_party/install/bin/:${PATH}"

declare -a args=(
    '-v'
    '--cov=torchaudio'
    "--junitxml=${PWD}/test-results/junit.xml"
    '--durations' '20'
)

cd test
pytest "${args[@]}" torchaudio_unittest
