Loading and saving audio and video
==================================

TorchAudio project has started as a simple audio IO library for PyTorch, which are
:py:func:`torchaudio.info`, :py:func:`torchaudio.load` and :py:func:`torchaudio.save`.

.. toctree::
   :maxdepth: 1

   tutorials/audio_io_tutorial

|

In recent releases, more powerful IO features were added in :py:mod:`torchaudio.io`.
These features are based on FFmpeg libraries, thus cross-platform and can handle
wide variety of media formats, including audio and video, coming from many different source.

The following tutorials shows how to use :py:class:`torchaudio.io.StreamReader` to
load audio and video from various sources.

.. toctree::
   :maxdepth: 1

   tutorials/streamreader_basic_tutorial
   tutorials/streamreader_advanced_tutorial

|

The following tutorials shows how to use :py:class:`torchaudio.io.StreamWriter` to
save and play audio and video.

.. toctree::
   :maxdepth: 1

   tutorials/streamwriter_basic_tutorial
   tutorials/streamwriter_advanced

|

:py:class:`~torchaudio.io.StreamReader` and :py:class:`~torchaudio.io.StreamWriter`
support GPU decoding and encoding. The following tutorial shows how to set up an
environment (i.e. how to install FFmpeg with NVDEC/NVENC support), and
use the GPU decoding/encoding from torchaudio.

.. toctree::
   :maxdepth: 1

   hw_acceleration_tutorial
