torchaudio
==========

.. currentmodule:: torchaudio

.. warning::
    Starting with version 2.8, we are refactoring TorchAudio to transition it
    into a maintenance phase. As a result:

    - The APIs listed below are deprecated in 2.8 and will be removed in 2.9.
    - The decoding and encoding capabilities of PyTorch for both audio and video
      are being consolidated into TorchCodec.

    Please see https://github.com/pytorch/audio/issues/3902 for more information.

I/O
---

``torchaudio`` top-level module provides the following functions that make
it easy to handle audio data.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/io.rst

   info
   load
   save
   list_audio_backends

.. _backend:

Backend and Dispatcher
----------------------

Decoding and encoding media is highly elaborated process. Therefore, TorchAudio
relies on third party libraries to perform these operations.
