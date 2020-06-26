#!/bin/bash
# Build third party libraries (SoX, lame, libmad, and flac)
# Usage: ./build_thid_parth.sh [prefix] [download_only?=false]

set -e

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
root_dir="${this_dir}/../.."

prefix="${1:-}"
if [ -z "${prefix}" ]; then
    prefix="${root_dir}"
fi
download_only="${2:-false}"

tp_dir="${prefix}/third_party"
tmp_dir="${tp_dir}/tmp"
build_dir="${tp_dir}/build"

mkdir -p "${tmp_dir}" "${build_dir}"

. "${this_dir}/build_third_party_helper.sh"

if ! found_ogg "${build_dir}" ; then
    get_ogg "${tmp_dir}"
    if [ "${download_only}" = "false" ]; then
        build_ogg "${tmp_dir}" "${build_dir}"
    fi
fi

if ! found_vorbis "${build_dir}" ; then
    get_vorbis "${tmp_dir}"
    if [ "${download_only}" = "false" ]; then
        build_vorbis "${tmp_dir}" "${build_dir}"
    fi
fi

if ! found_lame "${build_dir}" ; then
    get_lame "${tmp_dir}"
    if [ "${download_only}" = "false" ]; then
        build_lame "${tmp_dir}" "${build_dir}"
    fi
fi

if ! found_flac "${build_dir}" ; then
   get_flac "${tmp_dir}"
   if [ "${download_only}" = "false" ]; then
       build_flac "${tmp_dir}" "${build_dir}"
   fi
fi

if ! found_mad "${build_dir}" ; then
   get_mad "${tmp_dir}"
   if [ "${download_only}" = "false" ]; then
       build_mad "${tmp_dir}" "${build_dir}"
   fi
fi

if ! found_sox "${build_dir}" ; then
   get_sox "${tmp_dir}"
   if [ "${download_only}" = "false" ]; then
       build_sox "${tmp_dir}" "${build_dir}"
   fi
fi
