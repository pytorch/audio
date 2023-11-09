#!/usr/bin/env bash

set -euxo pipefail

eval "$(./conda/Scripts/conda.exe 'shell.bash' 'hook')"
conda activate ./env

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "$this_dir/set_cuda_envs.sh"

python -m torch.utils.collect_env
env | grep TORCHAUDIO || true

cd test
pytest --continue-on-collection-errors --cov=torchaudio --junitxml=${RUNNER_TEST_RESULTS_DIR}/junit.xml -v --durations 20 torchaudio_unittest
coverage html
