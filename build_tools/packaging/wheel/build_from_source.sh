rm -rf source_code
mkdir source_code
cd source_code

wget -q -O sox-14.4.2.tar.bz2 "http://downloads.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2.tar.bz2?r=http%3A%2F%2Fsourceforge.net%2Fprojects%2Fsox%2Ffiles%2Fsox%2F14.4.2%2F&ts=1416316415&use_mirror=heanet"
wget -q -O lame-3.99.5.tar.gz "http://downloads.sourceforge.net/project/lame/lame/3.99/lame-3.99.5.tar.gz?r=http%3A%2F%2Fsourceforge.net%2Fprojects%2Flame%2Ffiles%2Flame%2F3.99%2F&ts=1416316457&use_mirror=kent"
wget -q -O flac-1.3.2.tar.xz "https://superb-dca2.dl.sourceforge.net/project/flac/flac-src/flac-1.3.2.tar.xz"

# unpack the dependencies
tar xfp sox-14.4.2.tar.bz2
tar xfp lame-3.99.5.tar.gz
tar xfp flac-1.3.2.tar.xz

# build lame, statically
pushd lame-3.99.5
./configure --disable-shared --enable-static --prefix="$PREFIX/audio/third_party/lame" CFLAGS=-fPIC CXXFLAGS=-fPIC --with-pic --disable-debug --disable-dependency-tracking --enable-nasm
make -s && make install
popd

# build flac, statically
pushd flac-1.3.2
./configure --disable-shared --enable-static --prefix="$PREFIX/audio/third_party/flac" CFLAGS=-fPIC CXXFLAGS=-fPIC \
	--with-pic --disable-debug --disable-dependency-tracking
make -s && make install
popd

# build sox, statically
pushd sox-14.4.2
./configure --disable-shared --enable-static --prefix="$PREFIX/audio/third_party/sox" \
    LDFLAGS="-L$PREFIX/audio/third_party/lame/lib -L$PREFIX/audio/third_party/flac/lib" \
    CPPFLAGS="-I$PREFIX/audio/third_party/lame/include -I$PREFIX/audio/third_party/flac/include" \
    --with-lame --with-flac --without-oggvorbis --without-oss --without-sndfile CFLAGS=-fPIC CXXFLAGS=-fPIC --with-pic --disable-debug --disable-dependency-tracking
make -s && make install
popd
