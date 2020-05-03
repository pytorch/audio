#!/usr/bin/env bash

set -e

if [[ "$OSTYPE" == "msys" ]]; then
    eval "$(./conda/Scripts/conda.exe 'shell.bash' 'hook')"
else
    eval "$(./conda/bin/conda shell.bash hook)"
fi
conda activate ./env

python -m torch.utils.collect_env
pytest --cov=torchaudio --junitxml=test-results/junit.xml -v --durations 20 test
flake8 torchaudio test
