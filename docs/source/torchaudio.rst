torchaudio
==========

.. currentmodule:: torchaudio

.. warning::
    Starting with version 2.9, we have transitioned TorchAudio into a maintenance phase. As a result:

    - APIs deprecated in version 2.8 have been removed in 2.9.
    - The decoding and encoding capabilities of PyTorch for both audio and video
      have been consolidated into TorchCodec. For convenience,
      :func:`~torchaudio.load` and :func:`~torchaudio.save` are now aliases to
      :func:`~torchaudio.load_with_torchcodec` and :func:`~torchaudio.save_with_torchcodec`
      respectively, which call the appropriate functions from TorchCodec.
      That said, we recommend that you port your code to native torchcodec APIs.

    Please see https://github.com/pytorch/audio/issues/3902 for more information.

I/O
---

``torchaudio`` top-level module provides the following functions that make
it easy to handle audio data.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/io.rst

   load
   load_with_torchcodec
   save
   save_with_torchcodec
