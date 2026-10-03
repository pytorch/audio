#!/bin/bash

echo "Building torchaudio dependencies and wheel started."

export SRC_PATH="$GITHUB_WORKSPACE/$SRC_DIR"
export PYTORCH_VERSION="$PYTORCH_VERSION"
export CHANNEL="$CHANNEL"

# Source directory
cd "$SRC_PATH" || exit

# Create virtual environment
python -m pip install --upgrade pip
python -m venv .venv
echo "*" > .venv/.gitignore
source .venv/Scripts/activate

if [ "$CHANNEL" = "release" ]; then
  echo "Installing latest stable version of PyTorch."
  pip3 install --pre torch --index-url https://download.pytorch.org/whl/torch/
elif [ "$CHANNEL" = "test" ]; then
  echo "Installing PyTorch version $PYTORCH_VERSION."
  pip3 install --pre torch=="$PYTORCH_VERSION" --index-url https://download.pytorch.org/whl/test
else
  echo "CHANNEL is not set, installing PyTorch from nightly."
  pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
fi

echo "Dependencies install finished successfully."
