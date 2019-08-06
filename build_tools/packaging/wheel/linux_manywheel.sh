#!/bin/bash

set -ex

if [[ -z "$TORCHAUDIO_BUILD_VERSION" ]]; then
  export TORCHAUDIO_BUILD_VERSION="0.4.0.dev$(date "+%Y%m%d")"
fi
if [[ -z "$TORCHAUDIO_BUILD_NUMBER" ]]; then
  export TORCHAUDIO_BUILD_NUMBER="1"
fi
export OUT_DIR="/remote/cpu"

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd /opt/python
DESIRED_PYTHON=(*/)
for desired_py in "${DESIRED_PYTHON[@]}"; do
    python_installations+=("/opt/python/$desired_py")
done

if [[ "$TARGET_COMMIT" == HEAD ]]; then
  # Assume that this script was called from a valid checkout
  WORKDIR="$(realpath "$script_dir/../../..")"
else
  WORKDIR="/tmp/audio"
  cd /tmp
  rm -rf audio
  git clone https://github.com/pytorch/audio
  cd audio
  git checkout "$TARGET_COMMIT"
  git submodule update --init --recursive
fi

mkdir "$WORKDIR/third_party"

export PREFIX="$WORKDIR"
. "$script_dir/build_from_source.sh"

cd "$WORKDIR"

ORIG_TORCHAUDIO_PYTORCH_DEPENDENCY_VERSION="$TORCHAUDIO_PYTORCH_DEPENDENCY_VERSION"

OLD_PATH=$PATH
for PYDIR in "${python_installations[@]}"; do
    export PATH="$PYDIR/bin:$OLD_PATH"
    pip install --upgrade pip

    if [[ -z "$ORIG_TORCHAUDIO_PYTORCH_DEPENDENCY_VERSION" ]]; then
      pip install --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
      # CPU/CUDA variants of PyTorch have ABI compatible PyTorch.  Therefore, we
      # strip off the local package qualifier.  Also, we choose to build against
      # the CPU build, because it takes less time to download.
      export TORCHAUDIO_PYTORCH_DEPENDENCY_VERSION="$(pip show torch | grep ^Version: | sed 's/Version: \+//' | sed 's/+.\+//')"
    else
      pip install "torch==$TORCHAUDIO_PYTORCH_DEPENDENCY_VERSION" -f https://download.pytorch.org/whl/torch_stable.html
    fi
    echo "Building against ${TORCHAUDIO_PYTORCH_DEPENDENCY_VERSION}"

    # NB: do not actually install requirements.txt; that is only needed for
    # testing
    pip install numpy future
    IS_WHEEL=1 python setup.py clean
    IS_WHEEL=1 python setup.py bdist_wheel
    mkdir -p $OUT_DIR
    cp dist/*.whl $OUT_DIR/
done
