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

LAME="lame-3.99.5"
LAME_ARCHIVE="${LAME}.tar.gz"

get_lame() {
    work_dir="$1"
    url="https://downloads.sourceforge.net/project/lame/lame/3.99/${LAME_ARCHIVE}"
    (
        cd "${work_dir}"
        if [ ! -d "${LAME}" ]; then
            if [ ! -f "${LAME_ARCHIVE}" ]; then
                printf "Fetching liblame from %s\n" "${url}"
                curl $CURL_OPTS -O "${url}"
            fi
        fi
    )
}

build_lame() {
    work_dir="$1"
    install_dir="$2"
    (
        cd "${work_dir}"
        if [ ! -d "${LAME}" ]; then
            tar xfp "${LAME_ARCHIVE}"
        fi
        cd "${LAME}"
        # build statically
        printf "Building liblame\n"
        if [ ! -f Makefile ]; then
            ./configure ${CONFIG_OPTS} \
                        --disable-shared --enable-static --prefix="${install_dir}" CFLAGS=-fPIC CXXFLAGS=-fPIC \
                        --with-pic --disable-debug --disable-dependency-tracking --enable-nasm
        fi
        make ${MAKE_OPTS} > make.log 2>&1
        make ${MAKE_OPTS} install
    )
}

FLAC="flac-1.3.2"
FLAC_ARCHIVE="${FLAC}.tar.xz"

get_flac() {
    work_dir="$1"
    url="https://downloads.sourceforge.net/project/flac/flac-src/${FLAC_ARCHIVE}"
    (
        cd "${work_dir}"
        if [ ! -d "${FLAC}" ]; then
            if [ ! -f "${FLAC_ARCHIVE}" ]; then
                printf "Fetching flac from %s\n" "${url}"
                curl $CURL_OPTS -O "${url}"
            fi
        fi
    )
}

build_flac() {
    work_dir="$1"
    install_dir="$2"
    (
        cd "${work_dir}"
        if [ ! -d "${FLAC}" ]; then
            tar xfp "${FLAC_ARCHIVE}"
        fi
        cd "${FLAC}"
        # build statically
        printf "Building flac\n"
        if [ ! -f Makefile ]; then
            ./configure ${CONFIG_OPTS} \
                        --disable-shared --enable-static --prefix="${install_dir}" CFLAGS=-fPIC CXXFLAGS=-fPIC \
                        --with-pic --disable-debug --disable-dependency-tracking
        fi
        make ${MAKE_OPTS} > make.log 2>&1
        make ${MAKE_OPTS} install
    )
}

LIBMAD="libmad-0.15.1b"
LIBMAD_ARCHIVE="${LIBMAD}.tar.gz"

get_mad() {
    work_dir="$1"
    url="https://downloads.sourceforge.net/project/mad/libmad/0.15.1b/${LIBMAD_ARCHIVE}"
    (
        cd "${work_dir}"
        if [ ! -d "${LIBMAD}" ]; then
            if [ ! -f "${LIBMAD_ARCHIVE}" ]; then
                printf "Fetching mad from %s\n" "${url}"
                curl $CURL_OPTS -O "${url}"
            fi
        fi
    )
}

build_mad() {
    work_dir="$1"
    install_dir="$2"
    (
        cd "${work_dir}"
        if [ ! -d "${LIBMAD}" ]; then
            tar xfp "${LIBMAD_ARCHIVE}"
        fi
        cd "${LIBMAD}"
        # build statically
        printf "Building mad\n"
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

SOX="sox-14.4.2"
SOX_ARCHIVE="${SOX}.tar.bz2"

get_sox() {
    work_dir="$1"
    url="https://downloads.sourceforge.net/project/sox/sox/14.4.2/${SOX_ARCHIVE}"
    (
        cd "${work_dir}"
        if [ ! -d "${SOX}" ]; then
            if [ ! -f "${SOX_ARCHIVE}" ]; then
                printf "Fetching SoX from %s\n" "${url}"
                curl $CURL_OPTS -O "${url}"
            fi
        fi
    )
}

build_sox() {
    work_dir="$1"
    install_dir="$2"
    (
        cd "${work_dir}"
        if [ ! -d "${SOX}" ]; then
            tar xfp "${SOX_ARCHIVE}"
        fi
        cd "${SOX}"
        # build statically
        printf "Building SoX\n"
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
