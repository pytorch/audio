Additional dependencies:

`pip install flashlight-text`

Build and install [k2](https://k2-fsa.github.io/k2/installation/for_developers.html#build-a-release-version):

```
git clone git@github.com:k2-fsa/k2.git
cd k2
mkdir build_release
cd build_release
# mkdir build_debug
# cd build_debug

# select cuda version
export CUDA_HOME=/usr/local/cuda-11.8/
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export CUDA_TOOLKIT_ROOT=$CUDA_HOME
export CUDA_BIN_PATH=$CUDA_HOME
export CUDA_PATH=$CUDA_HOME
export CUDA_INC_PATH=$CUDA_HOME/targets/x86_64-linux
export CFLAGS=-I$CUDA_HOME/targets/x86_64-linux/include:$CFLAGS

which nvcc
nvcc --version

# https://k2-fsa.github.io/k2/installation/for_developers.html#build-a-release-version
cmake -DCMAKE_BUILD_TYPE=Release ..
# cmake -DCMAKE_BUILD_TYPE=Debug ..

make -j10

export PYTHONPATH=$PWD/../k2/python:$PYTHONPATH # for `import k2`
export PYTHONPATH=$PWD/lib:$PYTHONPATH # for `import _k2`

# To test that your build is successful, run
python3 -c "import k2; print(k2.__file__)"
python3 -c "import torch; import _k2; print(_k2.__file__)"
```

How to run:

```
n_nodes=1
exp_dir=./experiments
librispeech_path="path-to-your-librispeech-directory"
srun -p train --cpus-per-task=12 --gpus-per-node=8 --nodes $n_nodes --ntasks-per-node=8  \
  python train.py \
  --exp-dir $exp_dir \
  --librispeech-path $librispeech_path \
  --global-stats-path ./global_stats.json \
  --sp-model-path ./spm_unigram_1023.model \
  --epochs 200 \
  --nodes $n_nodes
```
