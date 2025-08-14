#!/usr/bin/env bash

# NOTE
# Currently Linux GPU code has separate run script hardcoded in GHA YAML file.
# Therefore the CUDA-related things in this script is not used, and it's broken.
# TODO: Migrate GHA Linux GPU test job to this script.

unset PYTORCH_VERSION
# No need to set PYTORCH_VERSION for unit test, as we use nightly PyTorch.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e

case "$(uname -s)" in
    Darwin*)
        os=MacOSX
        eval "$($(which conda) shell.bash hook)"
        ;;
    *)
        os=Linux
        eval "$("/opt/conda/bin/conda" shell.bash hook)"
esac

conda create -n ci -y python="${PYTHON_VERSION}"
conda activate ci

export GPU_ARCH_TYPE="cpu"

case $GPU_ARCH_TYPE in
  cpu)
    GPU_ARCH_ID="cpu"
    ;;
  cuda)
    VERSION_WITHOUT_DOT=$(echo "${GPU_ARCH_VERSION}" | sed 's/\.//')
    GPU_ARCH_ID="cu${VERSION_WITHOUT_DOT}"
    ;;
  *)
    echo "Unknown GPU_ARCH_TYPE=${GPU_ARCH_TYPE}"
    exit 1
    ;;
esac
PYTORCH_WHEEL_INDEX="https://download.pytorch.org/whl/${UPLOAD_CHANNEL}/${GPU_ARCH_ID}"
pip install --progress-bar=off --pre torch torchcodec --index-url="${PYTORCH_WHEEL_INDEX}"
pip install "numpy>=1.26"

# 2. Install torchaudio
conda install --quiet -y ninja cmake

printf "* Installing torchaudio\n"
export BUILD_CPP_TEST=1
pip install . -v --no-build-isolation

# 3. Install Test tools
printf "* Installing test tools\n"
# On this CI, for whatever reason, we're only able to install ffmpeg 4.
conda install -y "ffmpeg<5"

pip3 install parameterized requests coverage pytest pytest-cov scipy expecttest
