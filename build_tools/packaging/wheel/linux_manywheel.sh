#!/bin/bash

set -ex

export TORCHAUDIO_PACKAGE_NAME="torchaudio_nightly"
export TORCHAUDIO_BUILD_VERSION="0.4.0.dev$(date "+%Y%m%d")"
export TORCHAUDIO_BUILD_NUMBER="1"
export OUT_DIR="/remote/cpu"

cd /opt/python
DESIRED_PYTHON=(*/)
for desired_py in "${DESIRED_PYTHON[@]}"; do
    python_installations+=("/opt/python/$desired_py")
done

OLD_PATH=$PATH
cd /tmp
rm -rf audio
git clone https://github.com/pytorch/audio
mkdir audio/third_party

export PREFIX="/tmp/audio"
. /remote/wheel/build_from_source.sh

cd /tmp/audio

for PYDIR in "${python_installations[@]}"; do
    export PATH=$PYDIR/bin:$OLD_PATH
    pip install --upgrade pip

    # For true hermetic builds, you ought to be constructing the docker
    # from scratch each time.  But this makes things marginally safer if
    # you aren't doing this.
    pip uninstall -y torch || true
    pip uninstall -y torch_nightly || true

    export TORCHAUDIO_PYTORCH_DEPENDENCY_NAME=torch_nightly
    pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
    # CPU/CUDA variants of PyTorch have ABI compatible PyTorch.  Therefore, we
    # strip off the local package qualifier.  Also, we choose to build against
    # the CPU build, because it takes less time to download.
    export TORCHAUDIO_PYTORCH_DEPENDENCY_VERSION="$(pip show torch_nightly | grep ^Version: | sed 's/Version: \+//' | sed 's/+.\+//')"
    echo "Building against ${TORCHAUDIO_PYTORCH_DEPENDENCY_VERSION}"

    # NB: do not actually install requirements.txt; that is only needed for
    # testing
    pip install numpy future
    IS_WHEEL=1 python setup.py clean
    IS_WHEEL=1 python setup.py bdist_wheel
    mkdir -p $OUT_DIR
    cp dist/*.whl $OUT_DIR/
done
