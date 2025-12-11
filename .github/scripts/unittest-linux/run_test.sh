#!/usr/bin/env bash

set -e

eval "$($(which conda) shell.bash hook)"

conda activate ci

python -m torch.utils.collect_env
env | grep TORCHAUDIO || true


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
    export TORCHAUDIO_TEST_ALLOW_SKIP_IF_NO_CTC_DECODER=true
    export TORCHAUDIO_TEST_ALLOW_SKIP_IF_NO_MOD_unidecode=true
    export TORCHAUDIO_TEST_ALLOW_SKIP_IF_NO_MOD_inflect=true
    export TORCHAUDIO_TEST_ALLOW_SKIP_IF_NO_MOD_pytorch_lightning=true
    export TORCHAUDIO_TEST_ALLOW_SKIP_IF_NO_MULTIGPU_CUDA=true
    cd test
    pytest torchaudio_unittest -k "not torchscript and not fairseq and not demucs ${PYTEST_K_EXTRA}" 
)
