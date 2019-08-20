#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

 set -e

 echo 'List files from cached directories'
if [ -d $HOME/download ]; then
    echo 'download:'
    ls $HOME/download
fi
if [ -d $HOME/.cache/pip ]; then
    echo 'pip:'
    ls $HOME/.cache/pip
fi

 # Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead
deactivate

 # Add the miniconda bin directory to $PATH
export PATH=/home/travis/miniconda3/bin:$PATH
echo $PATH

 # Use the miniconda installer for setup of conda itself
pushd .
cd
mkdir -p download
cd download
if [[ ! -f /home/travis/miniconda3/bin/activate ]]
then
    if [[ ! -f miniconda.sh ]]
    then
        wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
             -O miniconda.sh
    fi
    chmod +x miniconda.sh && ./miniconda.sh -b -f
    conda update --yes conda
    echo "Creating environment to run tests in."
    conda create -n testenv --yes python="$PYTHON_VERSION"
fi
cd ..
popd

 # Activate the python environment we created.
source activate testenv

 # Install requirements via pip in our conda environment
pip install -r requirements.txt

 # Install the following only if running tests
if [[ "$SKIP_INSTALL" != "true" ]]; then
     # TorchAudio CPP Extensions
    python setup.py install
fi

if [[ "$RUN_EXAMPLE_TESTS" == "true" ]]; then
  # Install dependencies
  pip install sentencepiece PyAudio

  if [[ ! -d $HOME/download/fairseq ]]; then
    # Install fairseq from source
    git clone https://github.com/pytorch/fairseq $HOME/download/fairseq
    pushd $HOME/download/fairseq
    pip install --editable .
    popd
  fi

  mkdir -p $HOME/download/data
  # Install dictionary, sentence piece model, and model
  wget -nc -O $HOME/download/data/dict.txt https://download.pytorch.org/models/audio/dict.txt
  wget -nc -O $HOME/download/data/spm.model https://download.pytorch.org/models/audio/spm.model
  wget -nc -O $HOME/download/data/model.pt https://download.pytorch.org/models/audio/checkpoint_avg_60_80.pt
fi

echo "Finished installation"
