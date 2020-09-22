#!/usr/bin/env bash

set -e

# shellcheck source=../../../../tools/conda_envs/utils.sh
. "$(git rev-parse --show-toplevel)/tools/conda_envs/utils.sh"

init_conda
activate_env master "${PYTHON_VERSION}"

python -m torch.utils.collect_env
cd test
pytest --cov=torchaudio --junitxml=../test-results/junit.xml -v --durations 20 torchaudio_unittest
