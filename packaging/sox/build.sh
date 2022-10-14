#!/usr/bin/env bash                                                                                                                                                                


prefix="${SOX_ROOT}"
echo ${prefix}

build_dir=$(mktemp -d -t sox-build.XXXXXXXXXX)
cleanup() {
    rm -rf "${build_dir}"
}
trap 'cleanup $?' EXIT
cd "${build_dir}"

# yum -y install lame lame-devel
# yum -y install libvorbis-devel
# yum -y install opus opus-devel
# yum -y install opusfile opusfile-devel
# yum -y install flac-devel
# yum -y install libopencore-amr libopencore-amr-devel

wget https://sourceforge.net/projects/opencore-amr/files/opencore-amr/opencore-amr-0.1.5.tar.gz
tar -xf opencore-amr-0.1.5.tar.gz
cd opencore-amr-0.1.5
./configure --prefix="${prefix}"
make install
cd ..

wget https://downloads.sourceforge.net/project/lame/lame/3.99/lame-3.99.5.tar.gz
tar -xf lame-3.99.5.tar.gz
cd lame-3.99.5
./configure --prefix="${prefix}"
make install
cd ..

wget https://ftp.osuosl.org/pub/xiph/releases/ogg/libogg-1.3.3.tar.gz
tar -xf libogg-1.3.3.tar.gz
cd libogg-1.3.3
./configure --prefix="${prefix}"
make install
cd ..

wget https://ftp.osuosl.org/pub/xiph/releases/flac/flac-1.3.2.tar.xz
tar -xf flac-1.3.2.tar.xz
cd flac-1.3.2
./configure --prefix="${prefix}" --with-ogg --disable-cpplibs
make install
cd ..

wget https://ftp.osuosl.org/pub/xiph/releases/vorbis/libvorbis-1.3.6.tar.gz
tar -xf libvorbis-1.3.6.tar.gz
cd libvorbis-1.3.6
./configure --prefix="${prefix}" --with-ogg
make install
cd ..

wget https://ftp.osuosl.org/pub/xiph/releases/opus/opus-1.3.1.tar.gz
tar -xf opus-1.3.1.tar.gz
cd opus-1.3.1
./configure --prefix="${prefix}" --with-ogg
make install
cd ..

wget https://ftp.osuosl.org/pub/xiph/releases/opus/opusfile-0.12.tar.gz
tar -xf opusfile-0.12.tar.gz
cd opusfile-0.12
./configure --prefix="${prefix}" --disable-http
make install
cd ..

wget https://downloads.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2.tar.bz2
tar -xf sox-14.4.2.tar.bz2 # --strip-components 1
cd sox-14.4.2
./configure --prefix="${prefix}" \
# ./configure \
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
