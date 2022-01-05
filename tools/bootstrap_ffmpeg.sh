#!/usr/bin/env bash

# Helper script to install MINIMUM ffmpeg in centos:7.
# The goal of this script is to allow bootstrapping the ffmpeg-feature build
# for Linux/wheel build process which happens in centos-based Docker.
# It is not intended to build the useful feature subset of ffmpegs

set -eux

build_dir=$(mktemp -d -t ffmpeg-build-XXXXXXXXXX)
cleanup() {
    echo rm -rf "${build_dir}"
}
trap cleanup EXIT

cd "${build_dir}"

wget --quiet -O ffmpeg.tar.gz https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n4.1.8.tar.gz
tar -xf ffmpeg.tar.gz --strip-components 1
./configure \
    --disable-all \
    --disable-static \
    --enable-shared \
    --enable-pic \
    --disable-debug \
    --disable-doc \
    --disable-autodetect \
    --disable-x86asm \
    --enable-avcodec \
    --enable-avdevice \
    --enable-avfilter \
    --enable-avformat \
    --enable-avutil

make -j install
