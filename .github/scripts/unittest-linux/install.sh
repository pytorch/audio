#!/usr/bin/env bash

# NOTE
# Currently Linux GPU code has separate run script hardcoded in GHA YAML file.
# Therefore the CUDA-related things in this script is not used, and it's broken.
# TODO: Migrate GHA Linux GPU test job to this script.

unset PYTORCH_VERSION
# No need to set PYTORCH_VERSION for unit test, as we use nightly PyTorch.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e

case "$(uname -s)" in
    Darwin*)
        os=MacOSX
        eval "$($(which conda) shell.bash hook)"
        ;;
    *)
        os=Linux
        eval "$("/opt/conda/bin/conda" shell.bash hook)"
esac

conda create -n ci -y python="${PYTHON_VERSION}"
conda activate ci

# 1. Install PyTorch
if [ -z "${CUDA_VERSION:-}" ] ; then
    if [ "${os}" == MacOSX ] ; then
        cudatoolkit=''
    else
        cudatoolkit="cpuonly"
    fi
    version="cpu"
else
    version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"
    export CUDATOOLKIT_CHANNEL="nvidia"
    cudatoolkit="pytorch-cuda=${version}"
fi

printf "Installing PyTorch with %s\n" "${cudatoolkit}"
(
    if [ "${os}" == MacOSX ] ; then
      # TODO: this can be removed as soon as linking issue could be resolved
      #  see https://github.com/pytorch/pytorch/issues/62424 from details
      MKL_CONSTRAINT='mkl==2021.2.0'
      pytorch_build=pytorch
    else
      MKL_CONSTRAINT=''
      pytorch_build="pytorch[build="*${version}*"]"
    fi
    set -x

    if [[ -z "$cudatoolkit" ]]; then
        conda install ${CONDA_CHANNEL_FLAGS:-} -y -c "pytorch-${UPLOAD_CHANNEL}" $MKL_CONSTRAINT "pytorch-${UPLOAD_CHANNEL}::${pytorch_build}"
    else
        conda install pytorch ${cudatoolkit} ${CONDA_CHANNEL_FLAGS:-} -y -c "pytorch-${UPLOAD_CHANNEL}" -c nvidia  $MKL_CONSTRAINT
    fi
)

# 2. Install torchaudio
conda install --quiet -y ninja cmake

printf "* Installing torchaudio\n"
export BUILD_CPP_TEST=1
python setup.py install

# 3. Install Test tools
printf "* Installing test tools\n"
NUMBA_DEV_CHANNEL=""
if [[ "$(python --version)" = *3.9* || "$(python --version)" = *3.10* ]]; then
    # Numba isn't available for Python 3.9 and 3.10 except on the numba dev channel and building from source fails
    # See https://github.com/librosa/librosa/issues/1270#issuecomment-759065048
    NUMBA_DEV_CHANNEL="-c numba/label/dev"
fi
# Note: installing librosa via pip fail because it will try to compile numba.
(
    set -x
    conda install -y -c conda-forge ${NUMBA_DEV_CHANNEL} sox libvorbis 'librosa==0.10.0' parameterized 'requests>=2.20' 'ffmpeg>=6,<7'
    pip install kaldi-io SoundFile coverage pytest pytest-cov 'scipy==1.7.3' expecttest unidecode inflect Pillow sentencepiece pytorch-lightning 'protobuf<4.21.0' demucs tinytag pyroomacoustics flashlight-text git+https://github.com/kpu/kenlm
)
# Install fairseq
git clone https://github.com/pytorch/fairseq
cd fairseq
git checkout e47a4c8
pip install .
