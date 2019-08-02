if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. Pass package suffix."
    echo "Package suffix should be '' or '-cpu'"
    exit 1
fi
export PACKAGE_SUFFIX="$1" # '' or '-cpu'

export TORCHAUDIO_PACKAGE_NAME="torchaudio${PACKAGE_SUFFIX}"
export TORCHAUDIO_BUILD_VERSION="0.2.0"
export TORCHAUDIO_BUILD_NUMBER="1"
export OUT_DIR="/remote/$CUVER"

cd /opt/python
DESIRED_PYTHON=(*/)
for desired_py in "${DESIRED_PYTHON[@]}"; do
    python_installations+=("/opt/python/$desired_py")
done

OLD_PATH=$PATH
cd /tmp
rm -rf audio
git clone https://github.com/pytorch/audio -b pr/0.2-wheel-updates
mkdir audio/third_party

export PREFIX="/tmp"
. /remote/wheel/build_from_source.sh

cd /tmp/audio

for PYDIR in "${python_installations[@]}"; do
    export PATH=$PYDIR/bin:$OLD_PATH
    # For true hermetic builds, you ought to be constructing the docker
    # from scratch each time.  But this makes things marginally safer if
    # you aren't doing this.
    pip uninstall -y torch || true
    pip install --upgrade pip
    pip install numpy future torch
    # NB: do not actually install requirements.txt; that is only needed for
    # testing
    IS_WHEEL=1 python setup.py clean
    IS_WHEEL=1 python setup.py bdist_wheel
    mkdir -p $OUT_DIR
    cp dist/*.whl $OUT_DIR/
done
