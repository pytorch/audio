#!/usr/bin/env bash

set -e

eval "$(./conda/Scripts/conda.exe 'shell.bash' 'hook')"
conda activate ./env

python -m torch.utils.collect_env
mkdir artifacts
pytest --cov=torchaudio --junitxml=test-results/junit.xml -v --durations 20 test/test_librosa_compatibility.py::TestTransforms::test_basics3
flake8 torchaudio test
