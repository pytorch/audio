#!/usr/bin/env bash

set -e

case "$(uname -s)" in
    Darwin*) os=MacOSX;;
    *) os=Linux
esac


eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

python -m torch.utils.collect_env
export TORCHAUDIO_TEST_FAIL_IF_NO_EXTENSION=1
export PATH="${PWD}/third_party/install/bin/:${PATH}"

declare -a common_args=(
    '--cov=torchaudio'
    "--junitxml=${PWD}/test-results/junit.xml"
    '--durations' '20'
)
if [ "${os}" == MacOSX ] ; then
    declare -a args=('-q' '-n' 'auto' '--dist=loadscope')
else
    declare -a args=('-v')
fi
cd test
pytest "${args[@]}" "${common_args[@]}" torchaudio_unittest
