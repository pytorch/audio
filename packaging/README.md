# Building torchaudio packages for release

 ## Anaconda packages

 ### Linux

 ```bash
docker run -it --ipc=host --rm -v $(pwd):/remote soumith/conda-cuda bash
cd remote
PYTHON_VERSION=3.7 packaging/build_conda.sh
```

To install bz2,
```bash
cd /opt/conda/conda-bld/linux-64/
# install dependencies
conda install pytorch-cpu=1.1.0
conda install sox
# install torchaudio
conda install /opt/conda/conda-bld/linux-64/torchaudio-cpu-0.2.0-py27_1.tar.bz2
```

To upload bz2,
```bash
anaconda upload -u pytorch /opt/conda/conda-bld/linux-64/torchaudio*.bz2
```

 ### OSX

 ```bash
# create a fresh anaconda environment / install and activate it
PYTHON_VERSION=3.7 packaging/build_conda.sh
```

To install bz2,
```bash
cd /Users/jamarshon/anaconda3/conda-bld/osx-64/
# activate conda env (e.g
conda info --envs
conda activate /Users/jamarshon/minconda_wheel_env_tmp/envs/env2.7
# install dependencies
conda install pytorch-cpu=1.1.0
conda install sox
# install torchaudio
# and then try installing (e.g
conda install /Users/jamarshon/anaconda3/conda-bld/osx-64/torchaudio-0.2.0-py27_1.tar.bz2
```

To upload bz2,
```bash
anaconda upload -u pytorch /Users/jamarshon/anaconda3/conda-bld/osx-64/torchaudio*.bz2
```

 ## Wheels

 ### Linux

 ```bash
nvidia-docker run -it --ipc=host --rm -v $(pwd):/remote soumith/manylinux-cuda90:latest bash
cd remote
PYTHON_VERSION=3.7 packaging/build_wheel.sh
```

To install wheels,
```bash
cd ../cpu
/opt/python/cp35-cp35m/bin/pip install torchaudio-0.2-cp35-cp35m-linux_x86_64.whl
```

To upload wheels,
```bash
cd ../cpu
/opt/python/cp35-cp35m/bin/pip install twine
/opt/python/cp35-cp35m/bin/twine upload *.whl
```

 ### OSX

 ```bash
PYTHON_VERSION=3.7 packaging/build_wheel.sh
```

To install wheels,
```bash
cd ~/torchaudio_wheels
conda activate /Users/jamarshon/minconda_wheel_env_tmp/envs/env2.7
pip install torchaudio-0.2-cp27-cp27m-macosx_10_6_x86_64.whl
```

To upload wheels,
```bash
pip install twine
cd ~/torchaudio_wheels
twine upload *.whl
```
