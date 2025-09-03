#!/usr/bin/env bash

set -euxo pipefail

# eval "$(./conda/Scripts/conda.exe 'shell.bash' 'hook')"

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "$this_dir/set_cuda_envs.sh"

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"

conda run -p "${env_dir}" python -m torch.utils.collect_env
env | grep TORCHAUDIO || true

cd test
conda run -p "${env_dir}" pytest --continue-on-collection-errors --cov=torchaudio --junitxml=${RUNNER_TEST_RESULTS_DIR}/junit.xml -v --durations 20 torchaudio_unittest -k "not torchscript and not fairseq and not demucs and not librosa"
conda run -p "${env_dir}" coverage html
