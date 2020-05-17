#!/bin/bash

set -ex

# Arguments: PREFIX, specifying where to install dependencies into

PREFIX="$1"
echo $PREFIX

curl --retry 3 https://s3.amazonaws.com/ossci-windows/torchaudio_deps.7z --output /tmp/torchaudio_deps.7z
7z x /tmp/torchaudio_deps.7z -o"$PREFIX/third_party"
