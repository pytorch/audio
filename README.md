torchaudio: an audio library for PyTorch
========================================

[![Build Status](https://circleci.com/gh/pytorch/audio.svg?style=svg)](https://app.circleci.com/pipelines/github/pytorch/audio)
[![Documentation](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Ftorchaudio%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://pytorch.org/audio/)

The aim of torchaudio is to apply [PyTorch](https://github.com/pytorch/pytorch) to
the audio domain. By supporting PyTorch, torchaudio follows the same philosophy
of providing strong GPU acceleration, having a focus on trainable features through
the autograd system, and having consistent style (tensor names and dimension names).
Therefore, it is primarily a machine learning library and not a general signal
processing library. The benefits of PyTorch can be seen in torchaudio through
having all the computations be through PyTorch operations which makes it easy
to use and feel like a natural extension.

- [Support audio I/O (Load files, Save files)](http://pytorch.org/audio/stable/)
  - Load a variety of audio formats, such as `wav`, `mp3`, `ogg`, `flac`, `opus`, `sphere`, into a torch Tensor using SoX
  - [Kaldi (ark/scp)](http://pytorch.org/audio/stable/kaldi_io.html)
- [Dataloaders for common audio datasets](http://pytorch.org/audio/stable/datasets.html)
- Common audio transforms
    - [Spectrogram, AmplitudeToDB, MelScale, MelSpectrogram, MFCC, MuLawEncoding, MuLawDecoding, Resample](http://pytorch.org/audio/stable/transforms.html)
- Compliance interfaces: Run code using PyTorch that align with other libraries
    - [Kaldi: spectrogram, fbank, mfcc](https://pytorch.org/audio/stable/compliance.kaldi.html)

Dependencies
------------
* PyTorch (See below for the compatible versions)
* [optional] vesis84/kaldi-io-for-python commit cb46cb1f44318a5d04d4941cf39084c5b021241e or above

The following are the corresponding ``torchaudio`` versions and supported Python versions.

| ``torch``                | ``torchaudio``           | ``python``                      |
| ------------------------ | ------------------------ | ------------------------------- |
| ``master`` / ``nightly`` | ``main`` / ``nightly``   | ``>=3.6``, ``<=3.9``            |
| ``1.9.1``                | ``0.9.1``                | ``>=3.6``, ``<=3.9``            |
| ``1.9.0``                | ``0.9.0``                | ``>=3.6``, ``<=3.9``            |
| ``1.8.0``                | ``0.8.0``                | ``>=3.6``, ``<=3.9``            |
| ``1.7.1``                | ``0.7.2``                | ``>=3.6``, ``<=3.9``            |
| ``1.7.0``                | ``0.7.0``                | ``>=3.6``, ``<=3.8``            |
| ``1.6.0``                | ``0.6.0``                | ``>=3.6``, ``<=3.8``            |
| ``1.5.0``                | ``0.5.0``                | ``>=3.5``, ``<=3.8``            |
| ``1.4.0``                | ``0.4.0``                | ``==2.7``, ``>=3.5``, ``<=3.8`` |


Installation
------------

### Binary Distributions

To install the latest version using anaconda, run:

```
conda install -c pytorch torchaudio
```

To install the latest pip wheels, run:

```
pip install torchaudio -f https://download.pytorch.org/whl/torch_stable.html
```

(If you do not have torch already installed, this will default to installing
torch from PyPI. If you need a different torch configuration, preinstall torch
before running this command.)

### Nightly build

Note that nightly build is built on PyTorch's nightly build. Therefore, you need to install the latest PyTorch when you use nightly build of torchaudio.

**pip**

```
pip install --pre torchaudio -f https://download.pytorch.org/whl/nightly/torch_nightly.html
```

**conda**

```
conda install -y -c pytorch-nightly torchaudio
```

### From Source

On non-Windows platforms, the build process builds libsox and codecs that torchaudio need to link to. It will fetch and build libmad, lame, flac, vorbis, opus, and libsox before building extension. This process requires `cmake` and `pkg-config`. libsox-based features can be disabled with `BUILD_SOX=0`.
The build process also builds the RNN transducer loss. This functionality can be disabled by setting the environment variable `BUILD_RNNT=0`.

```bash
# Linux
python setup.py install

# OSX
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install

# Windows
# We need to use the MSVC x64 toolset for compilation, with Visual Studio's vcvarsall.bat or directly with vcvars64.bat.
# These batch files are under Visual Studio's installation folder, under 'VC\Auxiliary\Build\'.
# More information available at:
#   https://docs.microsoft.com/en-us/cpp/build/how-to-enable-a-64-bit-visual-cpp-toolset-on-the-command-line?view=msvc-160#use-vcvarsallbat-to-set-a-64-bit-hosted-build-architecture
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 && set BUILD_SOX=0 && python setup.py install
# or
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" && set BUILD_SOX=0 && python setup.py install
```

This is known to work on linux and unix distributions such as Ubuntu and CentOS 7 and macOS.
If you try this on a new system and find a solution to make it work, feel free to share it by opening an issue.

Quick Usage
-----------

```python
import torchaudio

waveform, sample_rate = torchaudio.load('foo.wav')  # load tensor from file
torchaudio.save('foo_save.wav', waveform, sample_rate)  # save tensor to file
```

Backend Dispatch
----------------

By default in OSX and Linux, torchaudio uses SoX as a backend to load and save files.
The backend can be changed to [SoundFile](https://pysoundfile.readthedocs.io/en/latest/)
using the following. See [SoundFile](https://pysoundfile.readthedocs.io/en/latest/)
for installation instructions.

```python
import torchaudio
torchaudio.set_audio_backend("soundfile")  # switch backend

waveform, sample_rate = torchaudio.load('foo.wav')  # load tensor from file, as usual
torchaudio.save('foo_save.wav', waveform, sample_rate)  # save tensor to file, as usual
```

**Note**
- SoundFile currently does not support mp3.
- "soundfile" backend is not supported by TorchScript.

API Reference
-------------

API Reference is located here: http://pytorch.org/audio/

Contributing Guidelines
-----------------------

Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md)

Disclaimer on Datasets
----------------------

This is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a GitHub issue. Thanks for your contribution to the ML community!
