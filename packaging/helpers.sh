#!/usr/bin/env bash

_list_wheel_files() {
    unzip -l "$1" | awk '{print $4}'
}

# $1 = path to wheel
# $2 = pattern to grep for in wheel files
# If files matching $2 are found in the wheel, the function errors.
assert_not_in_wheel() {
    wheel_files=$(_list_wheel_files "$1")
    if grep -q "$2" <<< "$wheel_files"
    then
        echo "Found files in $1 that start with $2. Exiting!!"
        exit 1
    fi
}

# See assert_not_in_wheel
assert_in_wheel() {
    wheel_files=$(_list_wheel_files "$1")
    if ! grep -q "$2" <<< "$wheel_files"
    then
        echo "Did not find files in $1 that start with $2. Exiting!!"
        exit 1
    fi
}

assert_ffmpeg_not_installed() {
    if command -v "ffmpeg" &> /dev/null
    then
        echo "ffmpeg is installed, but it shouldn't! Exiting!!"
        exit 1
    fi
}
