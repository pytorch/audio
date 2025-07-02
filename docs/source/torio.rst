.. py:module:: torio

torio
=====

.. currentmodule:: torio.io

.. warning::
    Starting with version 2.8, we are refactoring TorchAudio to transition it
    into a maintenance phase. As a result:
    - ``torio`` is deprecated in 2.8 and will be removed in 2.9.
    - The decoding and encoding capabilities of PyTorch for both audio and video
      are being consolidated into TorchCodec.
    Please see https://github.com/pytorch/audio/issues/3902 for more information.

``torio`` is an alternative top-level module for I/O features. It is the extraction of the core implementation of I/O feature of ``torchaudio``.

If you want to use the multimedia processing features, but do not want to depend on the entire ``torchaudio`` package, you can use ``torio``.

.. note::

   Currently, ``torio`` is distributed alongside ``torchaudio``, and there is no stand-alone
   procedure to install ``torio`` only. Please refer to https://pytorch.org/get-started/locally/
   for the installation of ``torchaudio``.
