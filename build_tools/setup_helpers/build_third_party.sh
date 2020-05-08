#!/bin/bash
# Build third party libraries in `<repo_root>/third_party/build` or in `<given_prefix>/third_party/build`.

set -e

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
root_dir="${this_dir}/../.."

prefix="${1:-}"
if [ -z "${prefix}" ]; then
    prefix="${root_dir}"
fi

tp_dir="${prefix}/third_party"
tmp_dir="${tp_dir}/tmp"
build_dir="${tp_dir}/build"

mkdir -p "${tmp_dir}" "${build_dir}"

. "${this_dir}/build_third_party_helper.sh"

if ! found_lame "${build_dir}" ; then
   build_lame "${tmp_dir}" "${build_dir}"
fi

if ! found_flac "${build_dir}" ; then
    build_flac "${tmp_dir}" "${build_dir}"
fi

if ! found_mad "${build_dir}" ; then
    build_mad "${tmp_dir}" "${build_dir}"
fi

if ! found_sox "${build_dir}" ; then
    build_sox "${tmp_dir}" "${build_dir}"
fi
