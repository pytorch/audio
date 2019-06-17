if [[ ":$PATH:" == *"conda"* ]]; then
    echo "existing anaconda install in PATH, remove it and run script"
    exit 1
fi
# download and activate anaconda
rm -rf ~/minconda_wheel_env_tmp
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh && \
    chmod +x Miniconda3-latest-MacOSX-x86_64.sh && \
    ./Miniconda3-latest-MacOSX-x86_64.sh -b -p ~/minconda_wheel_env_tmp && \
    rm Miniconda3-latest-MacOSX-x86_64.sh

 . ~/minconda_wheel_env_tmp/bin/activate


export TORCHAUDIO_BUILD_VERSION="0.2.0"
export TORCHAUDIO_BUILD_NUMBER="1"
export OUT_DIR=~/torchaudio_wheels

export MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++

pushd /tmp
rm -rf audio
git clone https://github.com/pytorch/audio -b v${TORCHAUDIO_BUILD_VERSION}
pushd audio

brew install sox libomp

desired_pythons=( "2.7" "3.5" "3.6" "3.7" )
# for each python
for desired_python in "${desired_pythons[@]}"
do
    # create and activate python env
    env_name="env$desired_python"
    conda create -yn $env_name python="$desired_python"
    conda activate $env_name

     # install torchaudio dependencies
    pip install -r requirements.txt

    python setup.py clean
    python setup.py bdist_wheel
    mkdir -p $OUT_DIR
    cp dist/*.whl $OUT_DIR/
done
popd
popd
