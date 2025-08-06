.. py:module:: torchaudio.io

torchaudio.io
=============

.. warning::
    Starting with version 2.8, we are refactoring TorchAudio to transition it
    into a maintenance phase. As a result:

    - The ``torchaudio.io`` module is deprecated in 2.8 and will be removed in 2.9.
    - The decoding and encoding capabilities of PyTorch for both audio and video
      are being consolidated into TorchCodec.

    Please see https://github.com/pytorch/audio/issues/3902 for more information.

.. currentmodule:: torchaudio.io

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/io_class.rst

   StreamReader
   StreamWriter
   AudioEffector
   play_audio

.. rubric:: Tutorials using ``torchaudio.io``

.. minigallery:: torchaudio.io
