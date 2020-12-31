#!/usr/bin/env bash

declare -a TORCHAUDIO_VERSIONS=("0.6.0")
declare -a PYTHON_VERSIONS=("3.6" "3.7" "3.8")

export TORCHAUDIO_VERSIONS
export PYTHON_VERSIONS

export KALDI_ROOT="${KALDI_ROOT:-$HOME}"  # Just to disable warning emitted from kaldi_io

_this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
_root_dir="$(git rev-parse --show-toplevel)"
_conda_dir="${_root_dir}/conda"
case "$(uname -s)" in
    Darwin*) _os="MacOSX";;
    *) _os="Linux"
esac

install_conda() {
    if [ ! -d "${_conda_dir}" ]; then
        printf "* Installing conda\n"
        wget -nv -O miniconda.sh "http://repo.continuum.io/miniconda/Miniconda3-latest-${_os}-x86_64.sh"
        bash ./miniconda.sh -b -f -p "${_conda_dir}"
        rm miniconda.sh
    fi
}

init_conda() {
    eval "$("${_conda_dir}/bin/conda" shell.bash hook)"
}

get_name() {
    echo "${1}-py${2}"
}

get_env_dir() {
    echo "${_root_dir}/envs/$(get_name "$1" "$2")"
}

create_env() {
    env_dir="$(get_env_dir "$1" "$2")"
    if [ ! -d "${env_dir}" ]; then
        printf "* Creating environment torchaudio: %s, Python: %s\n" "$1" "$2"
        conda create -q --prefix "${env_dir}" -y python="$2"
    fi
}

activate_env() {
    printf "* Activating environment torchaudio: %s, Python: %s\n" "$1" "$2"
    conda activate "$(get_env_dir "$1" "$2")"
}

install_release() {
    printf "* Installing torchaudio: %s\n" "$1"
    conda install -y -q torchaudio="$1" packaging -c pytorch
    # packaging is required in test to validate the torchaudio version for dump
}

install_build_dependencies() {
    printf "* Installing torchaudio dependencies except PyTorch - (Python: %s)\n" "$1"
    conda env update -q --file "${_this_dir}/environment.yml" --prune
}

build_master() {
    printf "* Installing PyTorch (py%s)\n" "$1"
    conda install -y -q pytorch "cpuonly" -c pytorch-nightly
    printf "* Installing torchaudio\n"
    cd "${_root_dir}" || exit 1
    git submodule update --init --recursive
    BUILD_SOX=1 python setup.py clean install
}
