#!/usr/bin/env bash

# This script builds FFmpeg with NVIDIA Video codec SDK support.
#
# IMPORTANT:
# The resulting library files are non-distributable.
# Do not ship them. Do not use them for binary build.

set -eux

revision="${FFMPEG_VERSION:-6.0}"
codec_header_version="${CODEC_HEADER_VERSION:-n12.0.16.0}"
prefix="${PREFIX:-${CONDA_PREFIX}}"
ccap="${COMPUTE_CAPABILITY:-86}"

# Build FFmpeg with NVIDIA Video Codec SDK
# TODO cache this
(
    git clone --quiet https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
    cd nv-codec-headers
    git checkout ${codec_header_version}
    make PREFIX=${prefix} install
)

conda install \
      --quiet --yes \
      -c conda-forge \
      yasm x264 gnutls pkg-config lame libopus libvpx openh264 openssl x264

(
    wget -q https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n${revision}.tar.gz
    tar -xf n${revision}.tar.gz
    cd ./FFmpeg-n${revision}
    # Sometimes, the resulting FFmpeg binaries have development version number.
    # Setting `revision` variable ensures that it has the right version number.
    export revision=${revision}
    ./configure \
        --prefix="${prefix}" \
        --extra-cflags="-I${prefix}/include" \
        --extra-ldflags="-L${prefix}/lib" \
        --nvccflags="-gencode arch=compute_${ccap},code=sm_${ccap} -O2" \
        --disable-doc \
        --enable-rpath \
        --disable-static \
        --enable-protocol=https \
        --enable-gnutls \
        --enable-shared \
        --enable-gpl \
        --enable-nonfree \
        --enable-libmp3lame \
        --enable-libx264 \
        --enable-cuda-nvcc \
        --enable-nvenc \
        --enable-cuvid \
        --enable-nvdec
    make clean
    make -j > /dev/null 2>&1
    make install
    # test
    # src="https://download.pytorch.org/torchaudio/tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4"
    # ffmpeg -y -vsync 0 -hwaccel cuvid -hwaccel_output_format cuda -c:v h264_cuvid -resize 360x240 -i "${src}" -c:a copy -c:v h264_nvenc -b:v 5M test.mp4
)
