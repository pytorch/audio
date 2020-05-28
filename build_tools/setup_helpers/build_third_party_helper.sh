#!/usr/bin/env bash

set -euo pipefail

# Global options
CURL_OPTS="-L --retry 10 --connect-timeout 5 --max-time 180"
MAKE_OPTS="-j"
CONFIG_OPTS=""

if [ -z ${DEBUG+x} ]; then
    CURL_OPTS="${CURL_OPTS} --silent --show-error"
    MAKE_OPTS="${MAKE_OPTS} --quiet"
    CONFIG_OPTS="${CONFIG_OPTS} --quiet"
fi

all_found() {
    dir="$1"
    shift
    while [ "$#" -gt 0 ]; do
        if [ ! -f "${dir}/$1" ]; then
            return 1
        fi
        shift
    done
}


found_lame() {
    all_found "$1" 'include/lame/lame.h' 'lib/libmp3lame.a'
}

found_flac() {
    all_found "$1" \
              'include/FLAC/format.h' \
              'include/FLAC/stream_decoder.h' \
              'include/FLAC/export.h' \
              'include/FLAC/ordinals.h' \
              'include/FLAC/all.h' \
              'include/FLAC/assert.h' \
              'include/FLAC/callback.h' \
              'include/FLAC/metadata.h' \
              'include/FLAC/stream_encoder.h' \
              'include/FLAC++/export.h' \
              'include/FLAC++/decoder.h' \
              'include/FLAC++/all.h' \
              'include/FLAC++/metadata.h' \
              'include/FLAC++/encoder.h' \
              'lib/libFLAC++.a' \
              'lib/libFLAC.a'
}

found_mad() {
    all_found "$1" 'include/mad.h' 'lib/libmad.a'
}

found_sox() {
    all_found "$1" 'include/sox.h' 'lib/libsox.a'
}

build_lame() {
    work_dir="$1"
    install_dir="$2"
    package="lame-3.99.5"
    url="https://downloads.sourceforge.net/project/lame/lame/3.99/lame-3.99.5.tar.gz"
    (
        cd "${work_dir}"
        if [ ! -d "${package}" ]; then
            if [ ! -f "${package}.tar.gz" ]; then
                printf "Fetching liblame from %s\n" "${url}"
                curl $CURL_OPTS -o "${package}.tar.gz" "${url}"
            fi
            tar xfp "${package}.tar.gz"
        fi
        # build statically
        printf "Building liblame\n"
        cd "${package}"
        if [ ! -f Makefile ]; then
            ./configure ${CONFIG_OPTS} \
                        --disable-shared --enable-static --prefix="${install_dir}" CFLAGS=-fPIC CXXFLAGS=-fPIC \
                        LIBS=-ltinfo --with-pic --disable-debug --disable-dependency-tracking --enable-nasm
        fi
        make ${MAKE_OPTS} > make.log 2>&1
        make ${MAKE_OPTS} install
    )
}

build_flac() {
    work_dir="$1"
    install_dir="$2"
    package="flac-1.3.2"
    url="https://downloads.sourceforge.net/project/flac/flac-src/flac-1.3.2.tar.xz"
    (
        cd "${work_dir}"
        if [ ! -d "${package}" ]; then
            if [ ! -f "${package}.tar.xz" ]; then
                printf "Fetching flac from %s\n" "${url}"
                curl $CURL_OPTS -o "${package}.tar.xz" "${url}"
            fi
            tar xfp "${package}.tar.xz"
        fi
        # build statically
        printf "Building flac\n"
        cd "${package}"
        if [ ! -f Makefile ]; then
            ./configure ${CONFIG_OPTS} \
                        --disable-shared --enable-static --prefix="${install_dir}" CFLAGS=-fPIC CXXFLAGS=-fPIC \
                        --with-pic --disable-debug --disable-dependency-tracking
        fi
        make ${MAKE_OPTS} > make.log 2>&1
        make ${MAKE_OPTS} install
    )
}

build_mad() {
    work_dir="$1"
    install_dir="$2"
    package="libmad-0.15.1b"
    url="https://downloads.sourceforge.net/project/mad/libmad/0.15.1b/libmad-0.15.1b.tar.gz"
    (
        cd "${work_dir}"
        if [ ! -d "${package}" ]; then
            if [ ! -f "${package}.tar.gz" ]; then
                printf "Fetching mad from %s\n" "${url}"
                curl $CURL_OPTS -o "${package}.tar.gz" "${url}"
            fi
            tar xfp "${package}.tar.gz"
        fi
        # build statically
        printf "Building mad\n"
        cd "${package}"
        if [ ! -f Makefile ]; then
            # See https://stackoverflow.com/a/12864879/23845
            sed -i.bak 's/-march=i486//' configure
            ./configure ${CONFIG_OPTS} \
                        --disable-shared --enable-static --prefix="${install_dir}" CFLAGS=-fPIC CXXFLAGS=-fPIC \
                        --with-pic --disable-debug --disable-dependency-tracking
        fi
        make ${MAKE_OPTS} > make.log 2>&1
        make ${MAKE_OPTS} install
    )
}

build_sox() {
    work_dir="$1"
    install_dir="$2"
    package="sox-14.4.2"
    url="https://downloads.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2.tar.bz2"
    (
        cd "${work_dir}"
        if [ ! -d "${package}" ]; then
            if [ ! -f "${package}.tar.bz2" ]; then
                printf "Fetching SoX from %s\n" "${url}"
                curl $CURL_OPTS -o "${package}.tar.bz2" "${url}"
            fi
            tar xfp "${package}.tar.bz2"
        fi
        # build statically
        printf "Building sox\n"
        cd "${package}"
        if [ ! -f Makefile ]; then
            # --without-png makes OS X build less hazardous; somehow the build
            # finds png and enables it.  We don't want it; we'd need to package
            # it statically if we do.
            ./configure ${CONFIG_OPTS} --disable-shared --enable-static --prefix="${install_dir}" \
                        LDFLAGS="-L${install_dir}/lib" CPPFLAGS="-I${install_dir}/include" \
                        --with-lame --with-flac --with-mad --without-alsa --without-coreaudio \
                        --without-png --without-oggvorbis --without-oss --without-sndfile \
                        CFLAGS=-fPIC CXXFLAGS=-fPIC --with-pic --disable-debug --disable-dependency-tracking
        fi
        make ${MAKE_OPTS} > make.log 2>&1
        make ${MAKE_OPTS} install
    )
}
