#!/bin/bash
set -ex

echo "Running pre build script..."
export FFMPEG_ROOT=${PWD}/third_party/ffmpeg
if [[ ! -d ${FFMPEG_ROOT} ]]; then
    packaging/ffmpeg/build.sh
fi
echo FFMPEG_ROOT=${FFMPEG_ROOT}
