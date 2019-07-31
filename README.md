torchaudio: an audio library for PyTorch
================================================

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

Installation
------------

### Binaries

To install pip wheels of 0.2.0, select the appropriate pip wheel for
your version of Python:

```
# Wheels for Python 2 are NOT supported

# Python 3.5
pip3 install http://download.pytorch.org/whl/torchaudio-0.2-cp35-cp35m-linux_x86_64.whl

# Python 3.6
pip3 install http://download.pytorch.org/whl/torchaudio-0.2-cp36-cp36m-linux_x86_64.whl

# Python 3.7
pip3 install http://download.pytorch.org/whl/torchaudio-0.2-cp37-cp37m-linux_x86_64.whl
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
sound, sample_rate = torchaudio.load('foo.mp3')
torchaudio.save('foo_save.mp3', sound, sample_rate) # saves tensor to file
```

API Reference
-----------

API Reference is located here: http://pytorch.org/audio/
