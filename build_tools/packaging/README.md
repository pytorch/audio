# Building torchaudio packages for release

 ## Anaconda packages

 ### Linux

 ```bash
nvidia-docker run -it --ipc=host --rm -v $(pwd):/remote soumith/conda-cuda bash
pushd remote/conda
conda config --add channels pytorch
conda config --add channels conda-forge
./build_audio.sh
```

To install bz2,
```bash
cd /opt/conda/conda-bld/linux-64/
conda install /opt/conda/conda-bld/linux-64/torchaudio-cpu-0.2.0-py27_1.tar.bz2
```

To upload bz2,
```bash
anaconda upload -u pytorch /opt/conda/conda-bld/linux-64/torchaudio*.bz2
```

 ### OSX

 ```bash
# create a fresh anaconda environment / install and activate it
cd packaging/conda
conda install -y conda-build anaconda-client
conda config --add channels pytorch
conda config --add channels conda-forge
./build_audio.sh
```

To install bz2,
```bash
cd /Users/jamarshon/anaconda3/conda-bld/osx-64/
# activate conda env (e.g
conda info --envs
conda activate /Users/jamarshon/minconda_wheel_env_tmp/envs/env3.5
# and then try installing (e.g
conda install /Users/jamarshon/anaconda3/conda-bld/osx-64/torchaudio-0.2.0-py35_1.tar.bz2
```

To upload bz2,
```bash
anaconda upload -u pytorch /Users/jamarshon/anaconda3/conda-bld/osx-64/torchaudio*.bz2
```

 ## Wheels

 ### Linux

 ```bash
nvidia-docker run -it --ipc=host --rm -v $(pwd):/remote soumith/manylinux-cuda90:latest bash
cd remote/wheel
./linux_manywheel.sh cpu
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
pushd wheel
./osx_wheel.sh
```

To install wheels,
```bash
cd ~/torchaudio_wheels
conda activate /Users/jamarshon/minconda_wheel_env_tmp/envs/env2.7
/Users/jamarshon/minconda_wheel_env_tmp/envs/env2.7/bin/python -m pip install torchaudio-0.2-cp27-cp27m-macosx_10_6_x86_64.whl
```

To upload wheels,
```bash
pip install twine
cd ~/torchaudio_wheels
twine upload *.whl
```
