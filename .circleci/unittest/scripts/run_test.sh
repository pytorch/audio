#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

python -m torch.utils.collect_env
pytest --cov=torchaudio --junitxml=test-results/junit.xml -v test
flake8 torchaudio test
