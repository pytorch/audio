#!/usr/bin/env bash

list_executables() {
    # List up executables in the given directory
    find "$1" -type f -executable
}

list_kaldi_libraries() {
    # List up shared libraries used by executables found in the given directory ($1)
    # that reside in Kaldi directory ($2)
    while read file; do
        ldd "${file}" | grep -o "${2}.* ";
    done < <(list_executables "$1") | sort -u
}

set -euo pipefail

kaldi_root="$(realpath "$1")"
target_dir="$(realpath "$2")"

bin_dir="${target_dir}/bin"
lib_dir="${target_dir}/lib"

mkdir -p "${bin_dir}" "${lib_dir}"

# 1. Copy featbins
printf "Copying executables to %s\n" "${bin_dir}"
while read file; do
    printf "  %s\n" "${file}"
    cp "${file}" "${bin_dir}"
done < <(list_executables "${kaldi_root}/src/featbin")

# 2. Copy dependent libraries from Kaldi
printf "Copying libraries to %s\n" "${lib_dir}"
while read file; do
    printf "  %s\n" "$file"
    # If it is not symlink, just copy to the target directory
    if [ ! -L "${file}" ]; then
        cp "${file}" "${lib_dir}"
        continue
    fi

    # If it is symlink,
    # 1. Copy the actual library to the target directory.
    library="$(realpath "${file}")"
    cp "${library}" "${lib_dir}"
    # 2. then if the name of the symlink is different from the actual library name,
    #    create the symlink in the target directory.
    lib_name="$(basename "${library}")"
    link_name="$(basename "${file}")"
    if [ "${lib_name}" != "${link_name}" ]; then
        printf "    Linking %s -> %s\n" "${lib_name}" "${link_name}"
        (
            cd "${lib_dir}"
            ln -sf "${lib_name}" "${link_name}"
        )
    fi
done < <(list_kaldi_libraries "${bin_dir}" "${kaldi_root}")
