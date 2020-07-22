#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

python -m torch.utils.collect_env
export PATH="${PWD}/third_party/install/bin/:${PATH}"
pytest -q -n auto --dist=loadscope --cov=torchaudio --junitxml=test-results/junit.xml --durations 20 test
