#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

pushd doc
pip install -r requirements.txt
make html
popd

