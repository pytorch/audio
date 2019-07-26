torchaudio: an audio library for PyTorch
========================================

[![Build Status](https://travis-ci.org/pytorch/audio.svg?branch=master)](https://travis-ci.org/pytorch/audio)

- [Support audio I/O (Load files, Save files)](http://pytorch.org/audio/)
  - Load the following formats into a torch Tensor
    - mp3, wav, aac, ogg, flac, avr, cdda, cvs/vms,
    - aiff, au, amr, mp2, mp4, ac3, avi, wmv,
    - mpeg, ircam and any other format supported by libsox.
    - [Kaldi (ark/scp)](http://pytorch.org/audio/kaldi_io.html)
- [Dataloaders for common audio datasets (VCTK, YesNo)](http://pytorch.org/audio/datasets.html)
- Common audio transforms
  - [Scale, PadTrim, DownmixMono, LC2CL, BLC2CBL, MuLawEncoding, MuLawExpanding](http://pytorch.org/audio/transforms.html)

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
sound, sample_rate = torchaudio.load('foo.mp3')
torchaudio.save('foo_save.mp3', sound, sample_rate) # saves tensor to file
```

API Reference
-------------

API Reference is located here: http://pytorch.org/audio/

Conventions
-----------

Torchaudio is standardized around the following naming conventions.

* waveform: a tensor of audio samples with shape (channel, time)
* sample_rate: the rate of audio samples (samples per second)
* specgram: a tensor of spectrogram with shape (channel, n_freqs, time)
* mel_specgram: a mel spectrogram with shape (channel, n_mels, time)
* hop_length: the number of samples between the starts of consecutive frames
* n_freqs: the number of bins in a linear spectrogram
* min_freq: the lowest frequency of the lowest band in a spectrogram
* max_freq: the highest frequency of the highest band in a spectrogram
* n_fft: the number of Fourier bins
* n_mfcc, n_mels: to be consistent with other similarly named variables, with shape (channel, n_mfcc, time) and (channel, n_mels, time)
* win_length: the length of the STFT window
* window_fn: for functions that creates windows e.g. torch.hann_window

Transforms expect the following shapes. In particular, the input of all transforms and functions assumes channel first.

* Spectrogram: (channel, time) -> (channel, n_freqs, time, 2)
* AmplitudeToDB: (channel, n_freqs, time, 2) -> (channel, n_freqs, time, 2)
* MelScale: (channel, time) -> (channel, n_mels, time)
* MelSpectrogram: (channel, time) -> (channel, n_mels, time, 2)
* MFCC: (channel, time) -> (channel, n_mfcc, time)
* MuLawEncode: (channel, time) -> (channel, time)
* MuLawDecode: (channel, time) -> (channel, time)
* Resample: (channel, time) -> (channel, time)
* STFT: (channel, time, 2) -> (channel, n_freqs, time, 2).
* ISTFT: (channel, n_freqs, time, 2) -> (channel, time, 2).
