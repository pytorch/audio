set -ex
source ./packaging/pkg_helpers.bash
export NO_CUDA_PACKAGE=1
setup_env 0.8.0
setup_wheel_python
setup_pip_pytorch_version
# pytorch is already installed
pip install --no-deps ~/workspace/torchaudio*

