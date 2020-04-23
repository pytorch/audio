#!/usr/bin/env bash

set -e

conda_location="${HOME}/miniconda3"
root_dir="$(git rev-parse --show-toplevel)"

eval "$(${conda_location}/bin/conda shell.bash hook)"
conda activate ./env

python -m torch.utils.collect_env
pytest -v test
