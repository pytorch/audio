Media IO
========

TorchAudio project has started as a simple audio IO library for PyTorch, which are
:py:func:`torchaudio.info`, :py:func:`torchaudio.load` and :py:func:`torchaudio.save`.

In recent releases, more powerful IO features were added in :py:mod:`torchaudio.io`.
These features are based on FFmpeg libraries, thus cross-platform and can handle
wide variety of media formats, including audio and video, coming from many different source.


.. toctree::
   :maxdepth: 1
   :caption: Media IO Tutorials

   tutorials/audio_io_tutorial
   tutorials/streamreader_basic_tutorial
   tutorials/streamreader_advanced_tutorial
   tutorials/streamwriter_basic_tutorial
   tutorials/streamwriter_advanced
   hw_acceleration_tutorial
