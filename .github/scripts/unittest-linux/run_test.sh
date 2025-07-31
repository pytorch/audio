#!/usr/bin/env bash

set -e

eval "$($(which conda) shell.bash hook)"

conda activate ci

python -m torch.utils.collect_env
env | grep TORCHAUDIO || true

export PATH="${PWD}/third_party/install/bin/:${PATH}"

declare -a args=(
    '--continue-on-collection-errors'
    '-v'
    '--cov=torchaudio'
    "--junitxml=${RUNNER_TEST_RESULTS_DIR}/junit.xml"
    '--durations' '20'
)

if [[ "${CUDA_TESTS_ONLY}" = "1" ]]; then
  args+=('-k' 'cuda or gpu')
fi

(
    cd build/temp*/test/cpp
    ctest
)

(
    cd test
    pytest torchaudio_unittest -k "not backend and not /io/ and not prototype and not sox and not ffmpeg and not fairseq and not hdemucs and not (torchscript and rnnt) and not torchscript_consistency"
)
