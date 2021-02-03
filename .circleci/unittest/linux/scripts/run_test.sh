#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

case "$(uname -s)" in
    Darwin*) os=MacOSX;;
    *) os=Linux
esac

python -m torch.utils.collect_env
if [ "${os}" == Linux ]; then
    cat /proc/cpuinfo
fi
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
