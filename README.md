torchaudio: an audio library for PyTorch
========================================

[![Build Status](https://travis-ci.org/pytorch/audio.svg?branch=master)](https://travis-ci.org/pytorch/audio)

The aim of torchaudio is to apply [PyTorch](https://github.com/pytorch/pytorch) to
the audio domain. By supporting PyTorch, torchaudio will follow the same philosophy
of providing strong GPU acceleration, having a focus on trainable features through
the autograd system, and having consistent style (tensor names and dimension names).
Therefore, it will be primarily a machine learning library and not a general signal
processing library. The benefits of Pytorch will be seen in torchaudio through
having all the computations be through Pytorch operations which makes it easy
to use and feel like a natural extension.

- [Support audio I/O (Load files, Save files)](http://pytorch.org/audio/)
  - Load the following formats into a torch Tensor
    - mp3, wav, aac, ogg, flac, avr, cdda, cvs/vms,
    - aiff, au, amr, mp2, mp4, ac3, avi, wmv,
    - mpeg, ircam and any other format supported by libsox.
    - [Kaldi (ark/scp)](http://pytorch.org/audio/kaldi_io.html)
- [Dataloaders for common audio datasets (VCTK, YesNo)](http://pytorch.org/audio/datasets.html)
- Common audio transforms
    - [Spectrogram, SpectrogramToDB, MelScale, MelSpectrogram, MFCC, MuLawEncoding, MuLawDecoding, Resample](http://pytorch.org/audio/transforms.html)
- Compliance interfaces: Run code using PyTorch that align with other libraries
    - [Kaldi: fbank, spectrogram, resample_waveform](https://pytorch.org/audio/compliance.kaldi.html)

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

With torchaudio being a machine learning library and built on top of PyTorch,
torchaudio is standardized around the following naming conventions. In particular,
tensors are assumed to have channel as the first dimension and time as the last
dimension (when applicable). This makes it consistent with PyTorch's dimensions.

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
* `window_fn`: for functions that creates windows e.g. torch.hann_window

Transforms expect the following dimensions.

* `Spectrogram`: (channel, time) -> (channel, freq, time)
* `AmplitudeToDB`: (channel, freq, time) -> (channel, freq, time)
* `MelScale`: (channel, time) -> (channel, mel, time)
* `MelSpectrogram`: (channel, time) -> (channel, mel, time)
* `MFCC`: (channel, time) -> (channel, mfcc, time)
* `MuLawEncode`: (channel, time) -> (channel, time)
* `MuLawDecode`: (channel, time) -> (channel, time)
* `Resample`: (channel, time) -> (channel, time)
