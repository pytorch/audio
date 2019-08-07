# A set of useful bash functions for common functionality we need to do in
# many build scripts

# Respecting PYTHON_VERSION and UNICODE_ABI, add (or install) the correct
# version of Python to perform a build.  Relevant to wheel builds.
setup_python() {
  if [[ "$(uname)" == Darwin ]]; then
    eval "$(conda shell.bash hook)"
    conda create -yn "env$PYTHON_VERSION" python="$PYTHON_VERSION"
    conda activate "env$PYTHON_VERSION"
  else
    case "$PYTHON_VERSION" in
      2.7)
        if [[ -n "$UNICODE_ABI" ]]; then
          python_abi=cp27-cp27mu
        else
          python_abi=cp27-cp27m
        fi
        ;;
      3.5) python_abi=cp35-cp35m ;;
      3.6) python_abi=cp36-cp36m ;;
      3.7) python_abi=cp37-cp37m ;;
    esac
    export PATH="/opt/python/$python_abi/bin:$PATH"
  fi
}

# Fill BUILD_VERSION if it doesn't exist already with a nightly string
# Usage: setup_build_version 0.2
setup_build_version() {
  if [[ -z "$BUILD_VERSION" ]]; then
    export BUILD_VERSION="$1.dev$(date "+%Y%m%d")"
  fi
}

# Set some useful variables for OS X, if applicable
setup_macos() {
  if [[ "$(uname)" == Darwin ]]; then
    export MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++
  fi
}

# Function to retry functions that sometimes timeout or have flaky failures
retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

# Install with pip a bit more robustly than the default
pip_install() {
  retry pip install --progress-bar off "$@"
}

# Install torch with pip, respecting PYTORCH_VERSION, and record the installed
# version into PYTORCH_VERSION, if applicable
setup_pip_pytorch_version() {
  if [[ -z "$PYTORCH_VERSION" ]]; then
    # Install latest prerelease CPU version of torch, per our nightlies.
    pip_install --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
    # CPU/CUDA variants of PyTorch have ABI compatible PyTorch.  Therefore, we
    # strip off the local package qualifier.
    export PYTORCH_VERSION="$(pip show torch | grep ^Version: | sed 's/Version:  *//' | sed 's/+.\+//')"
  else
    # TODO: Maybe add staging too
    pip_install "torch==$PYTORCH_VERSION" \
      -f https://download.pytorch.org/whl/torch_stable.html
  fi
}

# Fill PYTORCH_VERSION with the latest conda nightly version, and
# CONDA_CHANNEL_FLAGS with appropriate flags to retrieve these versions
setup_conda_pytorch_version() {
  if [[ -z "$PYTORCH_VERSION" ]]; then
    export CONDA_CHANNEL_FLAGS="-c pytorch-nightly"
    export PYTORCH_VERSION=$(conda search --json 'pytorch[channel=pytorch-nightly]' | python -c "import sys, json, re; print(re.sub(r'\\+.*$', '', json.load(sys.stdin)['pytorch'][-1]['version'])")
  else
    export CONDA_CHANNEL_FLAGS="-c pytorch"
  fi
}

# Translate CUDA_VERSION into CUDA_CUDATOOLKIT_CONSTRAINT
setup_conda_cudatoolkit_constraint() {
  export CONDA_CUDATOOLKIT_CONSTRAINT=""
}
