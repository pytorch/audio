torchaudio: an audio library for PyTorch
========================================

[![Build Status](https://travis-ci.org/pytorch/audio.svg?branch=master)](https://travis-ci.org/pytorch/audio)

The aim of torchaudio is to apply [PyTorch](https://github.com/pytorch/pytorch) to
the audio domain. By supporting PyTorch, torchaudio follows the same philosophy
of providing strong GPU acceleration, having a focus on trainable features through
the autograd system, and having consistent style (tensor names and dimension names).
Therefore, it is primarily a machine learning library and not a general signal
processing library. The benefits of Pytorch is be seen in torchaudio through
having all the computations be through Pytorch operations which makes it easy
to use and feel like a natural extension.

- [Support audio I/O (Load files, Save files)](http://pytorch.org/audio/)
  - Load the following formats into a torch Tensor using SoX
    - mp3, wav, aac, ogg, flac, avr, cdda, cvs/vms,
    - aiff, au, amr, mp2, mp4, ac3, avi, wmv,
    - mpeg, ircam and any other format supported by libsox.
    - [Kaldi (ark/scp)](http://pytorch.org/audio/kaldi_io.html)
- [Dataloaders for common audio datasets (VCTK, YesNo)](http://pytorch.org/audio/datasets.html)
- Common audio transforms
    - [Spectrogram, AmplitudeToDB, MelScale, MelSpectrogram, MFCC, MuLawEncoding, MuLawDecoding, Resample](http://pytorch.org/audio/transforms.html)
- Compliance interfaces: Run code using PyTorch that align with other libraries
    - [Kaldi: spectrogram, fbank, mfcc, resample_waveform](https://pytorch.org/audio/compliance.kaldi.html)

Dependencies
------------
* pytorch (nightly version needed for development)
* libsox v14.3.2 or above
* [optional] vesis84/kaldi-io-for-python commit cb46cb1f44318a5d04d4941cf39084c5b021241e or above

Quick install on
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

Installation
------------

### Binaries

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

At the moment, there is no automated nightly build process, but we occasionally
build nightlies based on PyTorch nightlies by hand following the instructions in
[packaging](packaging).  To install the latest nightly via pip, run:

```
pip install numpy
pip install torchaudio_nightly -f https://download.pytorch.org/whl/nightly/torch_nightly.html
```

To install the latest nightly via conda, run:

```
conda install -y -c pytorch-nightly torchaudio
```


### From Source

If your system configuration is not among the supported configurations
above, you can build from source.

```bash
# Linux
python setup.py install

# OSX
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install
```

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
assumed to have channel as the first dimension and time as the last
dimension (when applicable). This makes it consistent with PyTorch's dimensions.
For size names, the prefix `n_` is used (e.g. "a tensor of size (`n_freq`, `n_mel`)")
whereas dimension names do not have this prefix (e.g. "a tensor of
dimension (channel, time)")

* `waveform`: a tensor of audio samples with dimensions (channel, time)
* `sample_rate`: the rate of audio dimensions (samples per second)
* `specgram`: a tensor of spectrogram with dimensions (channel, freq, time)
* `mel_specgram`: a mel spectrogram with dimensions (channel, mel, time)
* `hop_length`: the number of samples between the starts of consecutive frames
* `n_fft`: the number of Fourier bins
* `n_mel`, `n_mfcc`: the number of mel and MFCC bins
* `n_freq`: the number of bins in a linear spectrogram
* `min_freq`: the lowest frequency of the lowest band in a spectrogram
* `max_freq`: the highest frequency of the highest band in a spectrogram
* `win_length`: the length of the STFT window
* `window_fn`: for functions that creates windows e.g. `torch.hann_window`

Transforms expect and return the following dimensions.

* `Spectrogram`: (channel, time) -> (channel, freq, time)
* `AmplitudeToDB`: (channel, freq, time) -> (channel, freq, time)
* `MelScale`: (channel, freq, time) -> (channel, mel, time)
* `MelSpectrogram`: (channel, time) -> (channel, mel, time)
* `MFCC`: (channel, time) -> (channel, mfcc, time)
* `MuLawEncode`: (channel, time) -> (channel, time)
* `MuLawDecode`: (channel, time) -> (channel, time)
* `Resample`: (channel, time) -> (channel, time)
* `Fade`: (channel, time) -> (channel, time)
* `Vol`: (channel, time) -> (channel, time)

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
