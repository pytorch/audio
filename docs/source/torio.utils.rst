.. py:module:: torio.utils

torio.utils
===========

``torio.utils`` module contains utility functions to query and configure the global state of third party libraries.

.. warning::
    Starting with version 2.8, we are refactoring TorchAudio to transition it
    into a maintenance phase. As a result:
    - ``torio`` is deprecated in 2.8 and will be removed in 2.9.
    - The decoding and encoding capabilities of PyTorch for both audio and video
      are being consolidated into TorchCodec.
    Please see https://github.com/pytorch/audio/issues/3902 for more information.

.. currentmodule:: torio.utils

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/utils.rst

   ffmpeg_utils
