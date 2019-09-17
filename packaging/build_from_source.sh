#!/bin/bash

set -ex

# Arguments: PREFIX, specifying where to install dependencies into

PREFIX="$1"

rm -rf /tmp/torchaudio-deps
mkdir /tmp/torchaudio-deps
pushd /tmp/torchaudio-deps


# Curl Settings
# 3 minutes is the absolute max for the curl command
# Retry up to 10 times, wait to connect at most 5s per time
CURL_OPTS=" --retry 10 --connect-timeout 5 --max-time 180 "

curl -L $CURL_OPTS -o sox-14.4.2.tar.bz2 "http://downloads.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2.tar.bz2"
curl -L $CURL_OPTS -o lame-3.99.5.tar.gz "http://downloads.sourceforge.net/project/lame/lame/3.99/lame-3.99.5.tar.gz"
curl -L $CURL_OPTS -o flac-1.3.2.tar.xz "https://superb-dca2.dl.sourceforge.net/project/flac/flac-src/flac-1.3.2.tar.xz"
curl -L $CURL_OPTS -o libmad-0.15.1b.tar.gz "https://downloads.sourceforge.net/project/mad/libmad/0.15.1b/libmad-0.15.1b.tar.gz"

# unpack the dependencies
tar xfp sox-14.4.2.tar.bz2
tar xfp lame-3.99.5.tar.gz
tar xfp flac-1.3.2.tar.xz
tar xfp libmad-0.15.1b.tar.gz

# build lame, statically
pushd lame-3.99.5
./configure --disable-shared --enable-static --prefix="$PREFIX/third_party/lame" CFLAGS=-fPIC CXXFLAGS=-fPIC --with-pic --disable-debug --disable-dependency-tracking --enable-nasm
make -s -j && make install
popd

# build flac, statically
pushd flac-1.3.2
./configure --disable-shared --enable-static --prefix="$PREFIX/third_party/flac" CFLAGS=-fPIC CXXFLAGS=-fPIC \
    --with-pic --disable-debug --disable-dependency-tracking
make -s -j && make install
popd

# build mad, statically
pushd libmad-0.15.1b
# See https://stackoverflow.com/a/12864879/23845
sed -i.bak 's/-march=i486//' configure
./configure --disable-shared --enable-static --prefix="$PREFIX/third_party/mad" CFLAGS=-fPIC CXXFLAGS=-fPIC \
    --with-pic --disable-debug --disable-dependency-tracking
make -s -j && make install
popd

# build sox, statically
# --without-png makes OS X build less hazardous; somehow the build
# finds png and enables it.  We don't want it; we'd need to package
# it statically if we do.
pushd sox-14.4.2
./configure --disable-shared --enable-static --prefix="$PREFIX/third_party/sox" \
    LDFLAGS="-L$PREFIX/third_party/lame/lib -L$PREFIX/third_party/flac/lib -L$PREFIX/third_party/mad/lib" \
    CPPFLAGS="-I$PREFIX/third_party/lame/include -I$PREFIX/third_party/flac/include -I$PREFIX/third_party/mad/include" \
    --with-lame --with-flac --with-mad --without-alsa --without-coreaudio --without-png --without-oggvorbis --without-oss --without-sndfile CFLAGS=-fPIC CXXFLAGS=-fPIC --with-pic --disable-debug --disable-dependency-tracking
make -s -j && make install
popd

popd
