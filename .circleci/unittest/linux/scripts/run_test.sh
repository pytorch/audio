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

if [ "${os}" == MacOSX ] ; then
    pytest -q -n auto --dist=loadscope --cov=torchaudio --junitxml=test-results/junit.xml --durations 20 test
else
    pytest -v --cov=torchaudio --junitxml=test-results/junit.xml --durations 20 test
fi
