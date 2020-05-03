#!/usr/bin/env bash

set -e

if [[ "$OSTYPE" == "msys" ]]; then
    eval "$(./conda/Scripts/conda.exe 'shell.bash' 'hook')"
else
    eval "$(./conda/bin/conda shell.bash hook)"
fi
conda activate ./env

codecov
