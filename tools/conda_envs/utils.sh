#!/usr/bin/env bash

declare -a TORCHAUDIO_VERSIONS=("0.6.0")
declare -a PYTHON_VERSIONS=("3.6" "3.7" "3.8")

export TORCHAUDIO_VERSIONS
export PYTHON_VERSIONS

export KALDI_ROOT="${KALDI_ROOT:-$HOME}"  # Just to disable warning emitted from kaldi_io

_root_dir="$(git rev-parse --show-toplevel)"
_this_dir="${_root_dir}/tools/conda_envs"
_conda_dir="${_root_dir}/conda"
case "$(uname -s)" in
    Darwin*) _os="MacOSX";;
    Linux*) _os="Linux";;
    *) _os="Windows"
esac

_install_conda_windows() {
    export tmp_conda="$(echo ${_conda_dir} | tr '/' '\\')"
    export miniconda_exe="$(echo ${_root_dir} | tr '/' '\\')\\miniconda.exe"
    curl --output miniconda.exe https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
    "${_this_dir}/install_conda.bat"
    rm miniconda.exe
    unset tmp_conda
    unset miniconda_exe
}

_install_conda() {
    wget -nv -O miniconda.sh "http://repo.continuum.io/miniconda/Miniconda3-latest-${_os}-x86_64.sh"
    bash ./miniconda.sh -b -f -p "${_conda_dir}"
    rm miniconda.sh
}

install_conda() {
    if [ ! -d "${_conda_dir}" ]; then
        printf "* Installing conda\n"
        if [ "${_os}" = "Windows" ] ; then
            _install_conda_windows
        else
            _install_conda
        fi
    fi
}

init_conda() {
    if [ "${_os}" = "Windows" ]; then
	script="${_conda_dir}/Scripts/conda.exe"
    else
	script="${_conda_dir}/bin/conda"
    fi
    eval "$("${script}" "shell.bash" 'hook')"
}

_print_env() {
    printf "torchaudio: %s, Python: %s, %s" "$1" "$2" "${CUDA_VERSION:-cpu}"
}

get_name() {
    echo "${1}-py${2}-${CUDA_VERSION:-cpu}"
}

get_env_dir() {
    echo "${_root_dir}/envs/$(get_name "$@")"
}

create_env() {
    env_dir="$(get_env_dir "$@")"
    if [ ! -d "${env_dir}" ]; then
        printf "* Creating environment %s\n" "$(_print_env "$@")"
        conda create -q --prefix "${env_dir}" -y python="$2"
    fi
}

activate_env() {
    printf "* Activating environment %s\n" "$(_print_env "$@")"
    conda activate "$(get_env_dir "$@")"
}

install_release() {
    printf "* Installing torchaudio: %s\n" "$1"
    conda install -y -q torchaudio="$1" packaging -c pytorch
    # packaging is required in test to validate the torchaudio version for dump
}

install_build_dependencies() {
    printf "* Installing torchaudio dependencies except PyTorch\n"
    conda env update -q --file "${_this_dir}/environment-${_os}.yml" --prune
}

build_master() {
    if [ -z "${CUDA_VERSION:-}" ] ; then
        cudatoolkit="cpuonly"
    else
        version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"
        cudatoolkit="cudatoolkit=${version}"
    fi

    printf "* Installing PyTorch %s\n" "${cudatoolkit}"
    conda install -y -q pytorch "${cudatoolkit}" -c pytorch-nightly
    printf "* Installing torchaudio\n"
    cd "${_root_dir}" || exit 1
    BUILD_SOX=1 python setup.py clean install
}
