torchaudio: an audio library for PyTorch
========================================

[![Build Status](https://circleci.com/gh/pytorch/audio.svg?style=svg)](https://app.circleci.com/pipelines/github/pytorch/audio)
[![Coverage](https://codecov.io/gh/pytorch/audio/branch/master/graph/badge.svg)](https://codecov.io/gh/pytorch/audio)
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
  - Load the following formats into a torch Tensor using SoX
    - mp3, wav, aac, ogg, flac, avr, cdda, cvs/vms,
    - aiff, au, amr, mp2, mp4, ac3, avi, wmv,
    - mpeg, ircam and any other format supported by libsox.
    - [Kaldi (ark/scp)](http://pytorch.org/audio/stable/kaldi_io.html)
- [Dataloaders for common audio datasets (VCTK, YesNo)](http://pytorch.org/audio/stable/datasets.html)
- Common audio transforms
    - [Spectrogram, AmplitudeToDB, MelScale, MelSpectrogram, MFCC, MuLawEncoding, MuLawDecoding, Resample](http://pytorch.org/audio/stable/transforms.html)
- Compliance interfaces: Run code using PyTorch that align with other libraries
    - [Kaldi: spectrogram, fbank, mfcc, resample_waveform](https://pytorch.org/audio/stable/compliance.kaldi.html)

Dependencies
------------
* PyTorch (See below for the compatible versions)
* libsox v14.3.2 or above (only required when building from source)
* [optional] vesis84/kaldi-io-for-python commit cb46cb1f44318a5d04d4941cf39084c5b021241e or above

The following are the corresponding ``torchaudio`` versions and supported Python versions.

| ``torch``                | ``torchaudio``           | ``python``                      |
| ------------------------ | ------------------------ | ------------------------------- |
| ``master`` / ``nightly`` | ``master`` / ``nightly`` | ``>=3.6``                       |
| ``1.7.0``                | ``0.7.0``                | ``>=3.6``                       |
| ``1.6.0``                | ``0.6.0``                | ``>=3.6``                       |
| ``1.5.0``                | ``0.5.0``                | ``>=3.5``                       |
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

Note that nightly build is build on PyTorch's nightly build. Therefore, you need to install the latest PyTorch when you use nightly build of torchaudio.

**pip**

```
pip install numpy
pip install --pre torchaudio -f https://download.pytorch.org/whl/nightly/torch_nightly.html
```

**conda**

```
conda install -y -c pytorch-nightly torchaudio
```

### From Source

If your system configuration is not among the supported configurations
above, you can build torchaudio from source.

This will require libsox v14.3.2 or above.

<Details><Summary>Click here for the examples on how to install SoX</Summary>

OSX (Homebrew):
```bash
brew install sox
```

Linux (Ubuntu):
```bash
sudo apt-get install sox libsox-dev libsox-fmt-all
```

Anaconda
```bash
conda install -c conda-forge sox
```

</Details>

```bash
# Linux
python setup.py install

# OSX
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install
```

Alternatively, the build process can build libsox and some optional codecs statically and torchaudio can link them, by setting environment variable `BUILD_SOX=1`.
The build process will fetch and build libmad, lame, flac, vorbis, opus, and libsox before building extension. This process requires `cmake` and `pkg-config`.

```bash
# Linux
BUILD_SOX=1 python setup.py install

# OSX
BUILD_SOX=1 MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install
```

This is known to work on linux and unix distributions such as Ubuntu and CentOS 7 and macOS.
If you try this on a new system and find a solution to make it work, feel free to share it by opening an issue.

#### Troubleshooting

<Details><Summary>checking build system type... ./config.guess: unable to guess system type</Summary>

Since the configuration file for codecs are old, they cannot correctly detect the new environments, such as Jetson Aarch. You need to replace the `config.guess` file in `./third_party/tmp/lame-3.99.5/config.guess` and/or `./third_party/tmp/libmad-0.15.1b/config.guess` with [the latest one](https://github.com/gcc-mirror/gcc/blob/master/config.guess).

See also: [#658](https://github.com/pytorch/audio/issues/658)

</Details>

<Details><Summary>Undefined reference to `tgetnum' when using `BUILD_SOX`</Summary>

If while building from within an anaconda environment you come across errors similar to the following:

```
../bin/ld: console.c:(.text+0xc1): undefined reference to `tgetnum'
```

Install `ncurses` from `conda-forge` before running `python setup.py install`:

```
# Install ncurses from conda-forge
conda install -c conda-forge ncurses
```

</Details>


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

Unlike SoX, SoundFile does not currently support mp3.

API Reference
-------------

API Reference is located here: http://pytorch.org/audio/

Conventions
-----------

With torchaudio being a machine learning library and built on top of PyTorch,
torchaudio is standardized around the following naming conventions. Tensors are
assumed to have channels as the first dimension and time as the last
dimension (when applicable). This makes it consistent with PyTorch's dimensions.
For size names, the prefix `n_` is used (e.g. "a tensor of size (`n_freq`, `n_mel`)")
whereas dimension names do not have this prefix (e.g. "a tensor of
dimension (channels, time)")

* `waveform`: a tensor of audio samples with dimensions (channels, time)
* `sample_rate`: the rate of audio dimensions (samples per second)
* `specgram`: a tensor of spectrogram with dimensions (channels, freq, time)
* `mel_specgram`: a mel spectrogram with dimensions (channels, mel, time)
* `hop_length`: the number of samples between the starts of consecutive frames
* `n_fft`: the number of Fourier bins
* `n_mel`, `n_mfcc`: the number of mel and MFCC bins
* `n_freq`: the number of bins in a linear spectrogram
* `min_freq`: the lowest frequency of the lowest band in a spectrogram
* `max_freq`: the highest frequency of the highest band in a spectrogram
* `win_length`: the length of the STFT window
* `window_fn`: for functions that creates windows e.g. `torch.hann_window`

Transforms expect and return the following dimensions.

* `Spectrogram`: (channels, time) -> (channels, freq, time)
* `AmplitudeToDB`: (channels, freq, time) -> (channels, freq, time)
* `MelScale`: (channels, freq, time) -> (channels, mel, time)
* `MelSpectrogram`: (channels, time) -> (channels, mel, time)
* `MFCC`: (channels, time) -> (channel, mfcc, time)
* `MuLawEncode`: (channels, time) -> (channels, time)
* `MuLawDecode`: (channels, time) -> (channels, time)
* `Resample`: (channels, time) -> (channels, time)
* `Fade`: (channels, time) -> (channels, time)
* `Vol`: (channels, time) -> (channels, time)

Complex numbers are supported via tensors of dimension (..., 2), and torchaudio provides `complex_norm` and `angle` to convert such a tensor into its magnitude and phase. Here, and in the documentation, we use an ellipsis "..." as a placeholder for the rest of the dimensions of a tensor, e.g. optional batching and channel dimensions.

Contributing Guidelines
-----------------------

Please let us know if you encounter a bug by filing an [issue](https://github.com/pytorch/audio/issues).

We appreciate all contributions. If you are planning to contribute back
bug-fixes, please do so without any further discussion.

If you plan to contribute new features, utility functions or extensions to the
core, please first open an issue and discuss the feature with us. Sending a PR
without discussion might end up resulting in a rejected PR, because we might be
taking the core in a different direction than you might be aware of.

Disclaimer on Datasets
----------------------

This is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a GitHub issue. Thanks for your contribution to the ML community!
