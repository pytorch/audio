#!/usr/bin/env bash
# Helper script for building codecs depending on libogg, such as libopus and opus.
# It is difficult to set environment variable inside of ExternalProject_Add,
# so this script sets necessary environment variables before running the given command

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
install_dir="${this_dir}/install"

export PKG_CONFIG_PATH="${install_dir}/lib/pkgconfig"
export LDFLAGS="-L${install_dir}/lib ${LDFLAGS}"
export CPPFLAGS="-I${install_dir}/include ${CPPFLAGS}"

$@
