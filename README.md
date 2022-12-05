torchaudio: an audio library for PyTorch
========================================

[![Build Status](https://circleci.com/gh/pytorch/audio.svg?style=svg)](https://app.circleci.com/pipelines/github/pytorch/audio)
[![Documentation](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Ftorchaudio%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://pytorch.org/audio/main/)
[![Anaconda Badge](https://anaconda.org/pytorch/torchaudio/badges/downloads.svg)](https://anaconda.org/pytorch/torchaudio)
[![Anaconda-Server Badge](https://anaconda.org/pytorch/torchaudio/badges/platforms.svg)](https://anaconda.org/pytorch/torchaudio)

![TorchAudio Logo](docs/source/_static/img/logo.png)

The aim of torchaudio is to apply [PyTorch](https://github.com/pytorch/pytorch) to
the audio domain. By supporting PyTorch, torchaudio follows the same philosophy
of providing strong GPU acceleration, having a focus on trainable features through
the autograd system, and having consistent style (tensor names and dimension names).
Therefore, it is primarily a machine learning library and not a general signal
processing library. The benefits of PyTorch can be seen in torchaudio through
having all the computations be through PyTorch operations which makes it easy
to use and feel like a natural extension.

- [Support audio I/O (Load files, Save files)](http://pytorch.org/audio/main/)
  - Load a variety of audio formats, such as `wav`, `mp3`, `ogg`, `flac`, `opus`, `sphere`, into a torch Tensor using SoX
  - [Kaldi (ark/scp)](http://pytorch.org/audio/main/kaldi_io.html)
- [Dataloaders for common audio datasets](http://pytorch.org/audio/main/datasets.html)
- Common audio transforms
    - [Spectrogram, AmplitudeToDB, MelScale, MelSpectrogram, MFCC, MuLawEncoding, MuLawDecoding, Resample](http://pytorch.org/audio/main/transforms.html)
- Compliance interfaces: Run code using PyTorch that align with other libraries
    - [Kaldi: spectrogram, fbank, mfcc](https://pytorch.org/audio/main/compliance.kaldi.html)

Dependencies
------------
* PyTorch (See below for the compatible versions)
* [optional] vesis84/kaldi-io-for-python commit cb46cb1f44318a5d04d4941cf39084c5b021241e or above

The following are the corresponding ``torchaudio`` versions and supported Python versions.

| | ``torch``                | ``torchaudio``           | ``python``                      |
| ----------- | ------------------------ | ------------------------ | ------------------------------- |
| Development | ``master`` / ``nightly`` | ``main`` / ``nightly``   | ``>=3.7``, ``<=3.10``            |
| Latest versioned release | ``1.13.0``               | ``0.13.0``               | ``>=3.7``, ``<=3.10``            |

<details><summary>Previous versions</summary>

| ``torch``                | ``torchaudio``           | ``python``                      |
| ------------------------ | ------------------------ | ------------------------------- |
| ``1.12.1``               | ``0.12.1``               | ``>=3.7``, ``<=3.10``           |
| ``1.12.0``               | ``0.12.0``               | ``>=3.7``, ``<=3.10``           |
| ``1.11.0``               | ``0.11.0``               | ``>=3.7``, ``<=3.9``            |
| ``1.10.0``               | ``0.10.0``               | ``>=3.6``, ``<=3.9``            |
| ``1.9.1``                | ``0.9.1``                | ``>=3.6``, ``<=3.9``            |
| ``1.9.0``                | ``0.9.0``                | ``>=3.6``, ``<=3.9``            |
| ``1.8.1``                | ``0.8.1``                | ``>=3.6``, ``<=3.9``            |
| ``1.8.0``                | ``0.8.0``                | ``>=3.6``, ``<=3.9``            |
| ``1.7.1``                | ``0.7.2``                | ``>=3.6``, ``<=3.9``            |
| ``1.7.0``                | ``0.7.0``                | ``>=3.6``, ``<=3.8``            |
| ``1.6.0``                | ``0.6.0``                | ``>=3.6``, ``<=3.8``            |
| ``1.5.0``                | ``0.5.0``                | ``>=3.5``, ``<=3.8``            |
| ``1.4.0``                | ``0.4.0``                | ``==2.7``, ``>=3.5``, ``<=3.8`` |

</details>

Installation
------------

### Binary Distributions

`torchaudio` has binary distributions for PyPI (`pip`) and Anaconda (`conda`).

Please refer to https://pytorch.org/get-started/locally/ for the details.

**Note** Starting `0.10`, torchaudio has CPU-only and CUDA-enabled binary distributions, each of which requires a matching PyTorch version.

**Note** This software was compiled against an unmodified copy of FFmpeg (licensed under [the LGPLv2.1](https://github.com/FFmpeg/FFmpeg/blob/a5d2008e2a2360d351798e9abe883d603e231442/COPYING.LGPLv2.1)), with the specific rpath removed so as to enable the use of system libraries. The LGPL source can be downloaded [here](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.1.8).

### From Source

On non-Windows platforms, the build process builds libsox and codecs that torchaudio need to link to. It will fetch and build libmad, lame, flac, vorbis, opus, and libsox before building extension. This process requires `cmake` and `pkg-config`. libsox-based features can be disabled with `BUILD_SOX=0`.
The build process also builds the RNN transducer loss and CTC beam search decoder. These functionalities can be disabled by setting the environment variable `BUILD_RNNT=0` and `BUILD_CTC_DECODER=0`, respectively.

```bash
# Linux
python setup.py install

# OSX
CC=clang CXX=clang++ python setup.py install

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

API Reference is located here: http://pytorch.org/audio/main/

Contributing Guidelines
-----------------------

Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md)

Citation
--------

If you find this package useful, please cite as:

```bibtex
@article{yang2021torchaudio,
  title={TorchAudio: Building Blocks for Audio and Speech Processing},
  author={Yao-Yuan Yang and Moto Hira and Zhaoheng Ni and Anjali Chourdia and Artyom Astafurov and Caroline Chen and Ching-Feng Yeh and Christian Puhrsch and David Pollack and Dmitriy Genzel and Donny Greenberg and Edward Z. Yang and Jason Lian and Jay Mahadeokar and Jeff Hwang and Ji Chen and Peter Goldsborough and Prabhat Roy and Sean Narenthiran and Shinji Watanabe and Soumith Chintala and Vincent Quenneville-Bélair and Yangyang Shi},
  journal={arXiv preprint arXiv:2110.15018},
  year={2021}
}
```

Disclaimer on Datasets
----------------------

This is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a GitHub issue. Thanks for your contribution to the ML community!
