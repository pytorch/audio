Load Audio files directly into PyTorch Tensors
================================================

Audio library for PyTorch
 * Support audio I/O (Load files)

Load the following formats into a torch Tensor
 * mp3, wav, aac, ogg, flac, avr, cdda, cvs/vms,
 * aiff, au, amr, mp2, mp4, ac3, avi, wmv,
 * mpeg, ircam and any other format supported by libsox.

Dependencies
------------
* libsox v14.3.2 or above

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

```bash
python setup.py install
```

Quick Usage
-----------

```python
import torchaudio
sound, sample_rate = torchaudio.load('foo.mp3')
```

API Reference
-----------
torchaudio.load
```
loads an audio file into a Tensor
audio.load(
	string,  # path to file
	out=None, # optionally pass output Tensor (any CPU Tensor type)
)
```

