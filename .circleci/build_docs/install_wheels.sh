#!/usr/bin/env bash

set -ex

if [[ -z "$PYTORCH_VERSION" ]]; then
    # Nightly build
    pip install --progress-bar off --pre torch -f "https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html"
else
    # Release branch
    pip install --progress-bar off "torch==${PYTORCH_VERSION}+cpu" \
        -f https://download.pytorch.org/whl/torch_stable.html \
        -f "https://download.pytorch.org/whl/${UPLOAD_CHANNEL}/torch_${UPLOAD_CHANNEL}.html"
fi
pip install --progress-bar off --no-deps ~/workspace/torchaudio*
pip install --progress-bar off -r docs/requirements.txt -r docs/requirements-tutorials.txt
