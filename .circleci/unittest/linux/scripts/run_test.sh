#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

python -m torch.utils.collect_env
export PATH="${PWD}/third_party/build/bin/:${PATH}"
pytest --cov=torchaudio --junitxml=test-results/junit.xml -v --durations 20 test
