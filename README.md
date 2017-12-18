Load Audio files directly into PyTorch Tensors
================================================

Audio library for PyTorch

- [Support audio I/O (Load files, Save files)](http://pytorch.org/audio/)
  - Load the following formats into a torch Tensor
    - mp3, wav, aac, ogg, flac, avr, cdda, cvs/vms,
    - aiff, au, amr, mp2, mp4, ac3, avi, wmv,
    - mpeg, ircam and any other format supported by libsox.
- [Dataloaders for common audio datasets (VCTK, YesNo)](http://pytorch.org/audio/datasets.html)
- Common audio transforms
  - [Scale, PadTrim, DownmixMono, LC2CL, BLC2CBL, MuLawEncoding, MuLawExpanding](http://pytorch.org/audio/transforms.html)

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
pip install cffi
python setup.py install
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
