#!/usr/bin/env bash

set -e


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

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

python -m torch.utils.collect_env
env | grep TORCHAUDIO || true

export PATH="${PWD}/third_party/install/bin/:${PATH}"

declare -a args=(
    '-v'
    '--cov=torchaudio'
    "--junitxml=${PWD}/test-results/junit.xml"
    '--durations' '20'
)

cd test
pytest "${args[@]}" torchaudio_unittest
coverage html
