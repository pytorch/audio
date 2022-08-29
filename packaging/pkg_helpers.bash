# A set of useful bash functions for common functionality we need to do in
# many build scripts


# Setup CUDA environment variables, based on CU_VERSION
#
# Inputs:
#   CU_VERSION (cpu, cu92, cu100)
#   NO_CUDA_PACKAGE (bool)
#   BUILD_TYPE (conda, wheel)
#
# Outputs:
#   VERSION_SUFFIX (e.g., "")
#   PYTORCH_VERSION_SUFFIX (e.g., +cpu)
#   WHEEL_DIR (e.g., cu100/)
#   CUDA_HOME (e.g., /usr/local/cuda-9.2, respected by torch.utils.cpp_extension)
#   USE_CUDA (respected by torchaudio setup.py)
#   NVCC_FLAGS (respected by torchaudio setup.py)
#
# Precondition: CUDA versions are installed in their conventional locations in
# /usr/local/cuda-*
#
# NOTE: Why VERSION_SUFFIX versus PYTORCH_VERSION_SUFFIX?  If you're building
# a package with CUDA on a platform we support CUDA on, VERSION_SUFFIX ==
# PYTORCH_VERSION_SUFFIX and everyone is happy.  However, if you are building a
# package with only CPU bits (e.g., torchaudio), then VERSION_SUFFIX is always
# empty, but PYTORCH_VERSION_SUFFIX is +cpu (because that's how you get a CPU
# version of a Python package.  But that doesn't apply if you're on OS X,
# since the default CU_VERSION on OS X is cpu.
setup_cuda() {

  # First, compute version suffixes.  By default, assume no version suffixes
  export VERSION_SUFFIX=""
  export PYTORCH_VERSION_SUFFIX=""
  export WHEEL_DIR="cpu/"
  # Wheel builds need suffixes (but not if they're on OS X, which never has suffix)
  if [[ "$BUILD_TYPE" == "wheel" ]] && [[ "$(uname)" != Darwin ]]; then
    export PYTORCH_VERSION_SUFFIX="+$CU_VERSION"
    # Match the suffix scheme of pytorch, unless this package does not have
    # CUDA builds (in which case, use default)
    if [[ -z "$NO_CUDA_PACKAGE" ]]; then
      export VERSION_SUFFIX="$PYTORCH_VERSION_SUFFIX"
      export WHEEL_DIR="$CU_VERSION/"
    fi
  fi

  # Now work out the CUDA settings
  case "$CU_VERSION" in
    cu117)
      if [[ "$OSTYPE" == "msys" ]]; then
        export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.7"
      else
        export CUDA_HOME=/usr/local/cuda-11.7/
      fi
      export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
      ;;
    cu116)
      if [[ "$OSTYPE" == "msys" ]]; then
        export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.6"
      else
        export CUDA_HOME=/usr/local/cuda-11.6/
      fi
      export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
      ;;
    cu113)
      if [[ "$OSTYPE" == "msys" ]]; then
        export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.3"
      else
        export CUDA_HOME=/usr/local/cuda-11.3/
      fi
      export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
      ;;
    cu102)
      if [[ "$OSTYPE" == "msys" ]]; then
        export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2"
      else
        export CUDA_HOME=/usr/local/cuda-10.2/
      fi
      export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5"
      ;;
    rocm*)
      export USE_ROCM=1
      ;;
    cpu)
      ;;
    *)
      echo "Unrecognized CU_VERSION=$CU_VERSION"
      exit 1
      ;;
  esac
  if [[ -n "$CUDA_HOME" ]]; then
    if [[ "$OSTYPE" == "msys" ]]; then
      export PATH="$CUDA_HOME\\bin:$PATH"
    else
      # Adds nvcc binary to the search path so that CMake's `find_package(CUDA)` will pick the right one
      export PATH="$CUDA_HOME/bin:$PATH"
    fi
    export USE_CUDA=1
  fi
}

# Populate build version if necessary, and add version suffix
#
# Inputs:
#   BUILD_VERSION (e.g., 0.2.0 or empty)
#   VERSION_SUFFIX (e.g., +cpu)
#
# Outputs:
#   BUILD_VERSION (e.g., 0.2.0.dev20190807+cpu)
#
# Fill BUILD_VERSION if it doesn't exist already with a nightly string
# Or retrieve it from the version.txt
# Usage: setup_build_version
setup_build_version() {
  if [[ -z "$BUILD_VERSION" ]]; then
    if [[ -z "$1" ]]; then
      setup_base_build_version
    else
      BUILD_VERSION="$1"
    fi
    BUILD_VERSION="$BUILD_VERSION.dev$(date "+%Y%m%d")$VERSION_SUFFIX"
  else
    BUILD_VERSION="$BUILD_VERSION$VERSION_SUFFIX"
  fi

  # Set build version based on tag if on tag
  if [[ -n "${CIRCLE_TAG}" ]]; then
    # Strip tag
    BUILD_VERSION="$(echo "${CIRCLE_TAG}" | sed -e 's/^v//' -e 's/-.*$//')${VERSION_SUFFIX}"
  fi

  export BUILD_VERSION
}

setup_base_build_version() {
  SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
  # version.txt for some reason has `a` character after major.minor.rev
  # command below yields 0.10.0 from version.txt containing 0.10.0a0
  BUILD_VERSION=$( cut -f 1 -d a "$SCRIPT_DIR/../version.txt" )
  export BUILD_VERSION
}

# Set some useful variables for OS X, if applicable
setup_macos() {
  if [[ "$(uname)" == Darwin ]]; then
    export CC=clang CXX=clang++
  fi
}

# Top-level entry point for things every package will need to do
#
# Usage: setup_env 0.2.0
setup_env() {
  # https://github.com/actions/checkout/issues/760#issuecomment-1097501613
  git config --global --add safe.directory /__w/audio/audio
  git submodule update --init --recursive
  setup_cuda
  setup_build_version
  setup_macos
}

# Function to retry functions that sometimes timeout or have flaky failures
retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

# Inputs:
#   PYTHON_VERSION (2.7, 3.5, 3.6, 3.7)
#   UNICODE_ABI (bool)
#
# Outputs:
#   PATH modified to put correct Python version in PATH
#
# Precondition: If Linux, you are in a soumith/manylinux-cuda* Docker image
setup_wheel_python() {
  if [[ "$(uname)" == Darwin || "$OSTYPE" == "msys" ]]; then
    eval "$(conda shell.bash hook)"
    conda env remove -n "env$PYTHON_VERSION" || true
    conda create -yn "env$PYTHON_VERSION" python="$PYTHON_VERSION"
    conda activate "env$PYTHON_VERSION"
    conda install --quiet -y pkg-config
  else
    case "$PYTHON_VERSION" in
      3.7) python_abi=cp37-cp37m ;;
      3.8) python_abi=cp38-cp38 ;;
      3.9) python_abi=cp39-cp39 ;;
      3.10) python_abi=cp310-cp310 ;;
      *)
        echo "Unrecognized PYTHON_VERSION=$PYTHON_VERSION"
        exit 1
        ;;
    esac
    export PATH="/opt/python/$python_abi/bin:$PATH"
  fi
}

# Install with pip a bit more robustly than the default
pip_install() {
  retry pip install --progress-bar off "$@"
}

# Install torch with pip, respecting PYTORCH_VERSION, and record the installed
# version into PYTORCH_VERSION, if applicable
setup_pip_pytorch_version() {
  if [[ -z "$PYTORCH_VERSION" ]]; then
    # Install latest prerelease version of torch, per our nightlies, consistent
    # with the requested cuda version
    pip_install --pre torch -f "https://download.pytorch.org/whl/nightly/${WHEEL_DIR}torch_nightly.html"
    # CUDA and CPU are ABI compatible on the CPU-only parts, so strip in this case
    export PYTORCH_VERSION="$(pip show torch | grep ^Version: | sed 's/Version:  *//' | sed 's/+.\+//')"
  else
    pip_install "torch==$PYTORCH_VERSION$PYTORCH_VERSION_SUFFIX" \
      -f https://download.pytorch.org/whl/torch_stable.html \
      -f "https://download.pytorch.org/whl/${UPLOAD_CHANNEL}/torch_${UPLOAD_CHANNEL}.html"
  fi
}

# Fill PYTORCH_VERSION with the latest conda nightly version, and
# CONDA_CHANNEL_FLAGS with appropriate flags to retrieve these versions
#
# You MUST have populated PYTORCH_VERSION_SUFFIX before hand.
setup_conda_pytorch_constraint() {
  CONDA_CHANNEL_FLAGS="${CONDA_CHANNEL_FLAGS}"
  if [[ -z "$PYTORCH_VERSION" ]]; then
    export CONDA_CHANNEL_FLAGS="${CONDA_CHANNEL_FLAGS} -c pytorch-nightly"
    if [[ "$OSTYPE" == "msys" ]]; then
      export PYTORCH_VERSION="$(conda search --json -c pytorch-nightly pytorch | python -c "import sys, json; data=json.load(sys.stdin); print(data['pytorch'][-1]['version'])")"
    else
      export PYTORCH_VERSION="$(conda search --json 'pytorch[channel=pytorch-nightly]' | python3 -c "import sys, json, re; print(re.sub(r'\\+.*$', '', json.load(sys.stdin)['pytorch'][-1]['version']))")"
    fi
  else
    export CONDA_CHANNEL_FLAGS="${CONDA_CHANNEL_FLAGS} -c pytorch -c pytorch-test -c pytorch-nightly"
  fi
  if [[ "$CU_VERSION" == cpu ]]; then
    export CONDA_PYTORCH_BUILD_CONSTRAINT="- pytorch==$PYTORCH_VERSION${PYTORCH_VERSION_SUFFIX}"
    export CONDA_PYTORCH_CONSTRAINT="- pytorch==$PYTORCH_VERSION"
  else
    export CONDA_PYTORCH_BUILD_CONSTRAINT="- pytorch==${PYTORCH_VERSION}${PYTORCH_VERSION_SUFFIX}"
    export CONDA_PYTORCH_CONSTRAINT="- pytorch==${PYTORCH_VERSION}${PYTORCH_VERSION_SUFFIX}"
  fi
  # TODO: Remove me later, see https://github.com/pytorch/pytorch/issues/62424 for more details
  if [[ "$(uname)" == Darwin ]]; then
    arch_name="$(uname -m)"
    if [ "${arch_name}" != "arm64" ]; then
      # Use less than equal to avoid version conflict in python=3.6 environment
      export CONDA_EXTRA_BUILD_CONSTRAINT="- mkl<=2021.2.0"
    fi
  fi
}

# Translate CUDA_VERSION into CUDA_CUDATOOLKIT_CONSTRAINT
setup_conda_cudatoolkit_constraint() {
  export CONDA_BUILD_VARIANT="cuda"
  if [[ "$(uname)" == Darwin ]]; then
    export CONDA_BUILD_VARIANT="cpu"
  else
    case "$CU_VERSION" in
      cu117)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- pytorch-cuda=11.7 # [not osx]"
        ;;
      cu116)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- pytorch-cuda=11.6 # [not osx]"
        ;;
      cu113)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=11.3,<11.4 # [not osx]"
        ;;
      cu102)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=10.2,<10.3 # [not osx]"
        ;;
      cpu)
        export CONDA_CUDATOOLKIT_CONSTRAINT=""
        export CONDA_BUILD_VARIANT="cpu"
        ;;
      *)
        echo "Unrecognized CU_VERSION=$CU_VERSION"
        exit 1
        ;;
    esac
  fi
}

# Build the proper compiler package before building the final package
setup_visual_studio_constraint() {
  if [[ "$OSTYPE" == "msys" ]]; then
      export VSTOOLCHAIN_PACKAGE=vs2019
      export VSDEVCMD_ARGS=''
      conda build $CONDA_CHANNEL_FLAGS --no-anaconda-upload packaging/$VSTOOLCHAIN_PACKAGE
      cp packaging/$VSTOOLCHAIN_PACKAGE/conda_build_config.yaml packaging/torchaudio/conda_build_config.yaml
  fi
}
