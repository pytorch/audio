#!/usr/bin/env bash

set -euxo pipefail

eval "$(./conda/Scripts/conda.exe 'shell.bash' 'hook')"

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "$this_dir/set_cuda_envs.sh"

conda run -p ./env python -m torch.utils.collect_env
env | grep TORCHAUDIO || true

cd test
conda run -p ./env pytest --continue-on-collection-errors --cov=torchaudio --junitxml=${RUNNER_TEST_RESULTS_DIR}/junit.xml -v --durations 20 torchaudio_unittest
conda run -p ./env coverage html
