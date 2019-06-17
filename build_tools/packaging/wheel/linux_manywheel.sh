if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. Pass cuda version"
    echo "CUDA version should be cu90, cu100 or cpu"
    exit 1
fi
export CUVER="$1" # cu90 cu100 cpu

export TORCHAUDIO_BUILD_VERSION="0.2.0"
export TORCHAUDIO_BUILD_NUMBER="1"
export OUT_DIR="/remote/$CUVER"

pushd /opt/python
DESIRED_PYTHON=(*/)
popd
for desired_py in "${DESIRED_PYTHON[@]}"; do
    python_installations+=("/opt/python/$desired_py")
done

OLD_PATH=$PATH
git clone https://github.com/pytorch/audio -b v${TORCHAUDIO_BUILD_VERSION}
yum -y install sox sox-devel
ln -s /usr/include/sox/sox.h /usr/include/sox.h
pushd audio
for PYDIR in "${python_installations[@]}"; do
    # wheels for numba does not work with python 2.7
    if [[ "$PYDIR" == "/opt/python/cp27-cp27m/" || "$PYDIR" == "/opt/python/cp27-cp27mu/" ]]; then
      continue;
    fi
    export PATH=$PYDIR/bin:$OLD_PATH
    pip install --upgrade pip
    pip install -r requirements.txt
    $PYDIR/bin/python setup.py clean
    $PYDIR/bin/python setup.py bdist_wheel
    mkdir -p $OUT_DIR
    cp dist/*.whl $OUT_DIR/
done
