#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e

root_dir="$(git rev-parse --show-toplevel)"
conda_dir="${root_dir}/conda"
env_dir="${root_dir}/env"

cd "${root_dir}"

case "$(uname -s)" in
    Darwin*) os=MacOSX;;
    *) os=Linux
esac


# if [ "${os}" != MacOSX ] ; then
    # apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
    # apt-get -qq update
    # apt-get -y install libopencore-amrnb-dev libopencore-amrwb-dev libflac-dev libvorbis-dev libmp3lame-dev opus-tools

wget https://sourceforge.net/projects/opencore-amr/files/opencore-amr/opencore-amr-0.1.5.tar.gz
tar -xf opencore-amr-0.1.5.tar.gz
cd opencore-amr-0.1.5
./configure
make install
cd ..

wget https://downloads.sourceforge.net/project/lame/lame/3.99/lame-3.99.5.tar.gz
tar -xf lame-3.99.5.tar.gz
cd lame-3.99.5
./configure
make install
cd ..

wget https://ftp.osuosl.org/pub/xiph/releases/ogg/libogg-1.3.3.tar.gz
tar -xf libogg-1.3.3.tar.gz
cd libogg-1.3.3
./configure
make install
cd ..

wget https://ftp.osuosl.org/pub/xiph/releases/flac/flac-1.3.2.tar.xz
tar -xf flac-1.3.2.tar.xz
cd flac-1.3.2
./configure --with-ogg --disable-cpplibs
make install
cd ..

wget https://ftp.osuosl.org/pub/xiph/releases/vorbis/libvorbis-1.3.6.tar.gz
tar -xf libvorbis-1.3.6.tar.gz
cd libvorbis-1.3.6
./configure --with-ogg
make install
cd ..

wget https://ftp.osuosl.org/pub/xiph/releases/opus/opus-1.3.1.tar.gz
tar -xf opus-1.3.1.tar.gz
cd opus-1.3.1
./configure
make install
cd ..

wget https://ftp.osuosl.org/pub/xiph/releases/opus/opusfile-0.12.tar.gz
tar -xf opusfile-0.12.tar.gz
cd opusfile-0.12
./configure --disable-http
make install
cd ..

wget https://downloads.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2.tar.bz2
tar -xf sox-14.4.2.tar.bz2
cd sox-14.4.2
./configure \
    --disable-openmp \
    --with-amrnb \
    --with-amrwb \
    --with-flac \
    --with-lame \
    --with-oggvorbis \
    --with-opus \
    --without-alsa \
    --without-ao \
    --without-coreaudio \
    --without-oss \
    --without-id3tag \
    --without-ladspa \
    --without-mad \
    --without-magic \
    --without-png \
    --without-pulseaudio \
    --without-sndfile \
    --without-sndio \
    --without-sunaudio \
    --without-waveaudio \
    --without-wavpack \
    --without-twolame
make install
cd ..
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"
# fi

# 0. Activate conda env
eval "$("${conda_dir}/bin/conda" shell.bash hook)"
conda activate "${env_dir}"

# 1. Install PyTorch
if [ -z "${CUDA_VERSION:-}" ] ; then
    if [ "${os}" == MacOSX ] ; then
        cudatoolkit=''
    else
        cudatoolkit="cpuonly"
    fi
    version="cpu"
else
    version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"
    export CUDATOOLKIT_CHANNEL="nvidia"
    cudatoolkit="pytorch-cuda=${version}"
fi

printf "Installing PyTorch with %s\n" "${cudatoolkit}"
(
    if [ "${os}" == MacOSX ] ; then
      # TODO: this can be removed as soon as linking issue could be resolved
      #  see https://github.com/pytorch/pytorch/issues/62424 from details
      MKL_CONSTRAINT='mkl==2021.2.0'
      pytorch_build=pytorch
    else
      MKL_CONSTRAINT=''
      pytorch_build="pytorch[build="*${version}*"]"
    fi
    set -x

    if [[ -z "$cudatoolkit" ]]; then
        conda install ${CONDA_CHANNEL_FLAGS:-} -y -c "pytorch-${UPLOAD_CHANNEL}" $MKL_CONSTRAINT "pytorch-${UPLOAD_CHANNEL}::${pytorch_build}"
    else
        conda install pytorch ${cudatoolkit} ${CONDA_CHANNEL_FLAGS:-} -y -c "pytorch-${UPLOAD_CHANNEL}" -c nvidia  $MKL_CONSTRAINT
    fi
)

# 2. Install torchaudio
printf "* Installing torchaudio\n"
python setup.py install

# 3. Install Test tools
printf "* Installing test tools\n"
NUMBA_DEV_CHANNEL=""
if [[ "$(python --version)" = *3.9* || "$(python --version)" = *3.10* ]]; then
    # Numba isn't available for Python 3.9 and 3.10 except on the numba dev channel and building from source fails
    # See https://github.com/librosa/librosa/issues/1270#issuecomment-759065048
    NUMBA_DEV_CHANNEL="-c numba/label/dev"
fi
# Note: installing librosa via pip fail because it will try to compile numba.
(
    set -x
    conda install -y -c conda-forge ${NUMBA_DEV_CHANNEL} 'librosa>=0.8.0' parameterized 'requests>=2.20'
    pip install kaldi-io SoundFile coverage pytest pytest-cov 'scipy==1.7.3' transformers expecttest unidecode inflect Pillow sentencepiece pytorch-lightning 'protobuf<4.21.0' demucs tinytag
)
# Install fairseq
git clone https://github.com/pytorch/fairseq
cd fairseq
git checkout e47a4c8
pip install .
