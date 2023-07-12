Installing pre-built binaries
=============================

``torchaudio`` has binary distributions for PyPI (``pip``) and Anaconda (``conda``).

Please refer to https://pytorch.org/get-started/locally/ for the details.

.. note::

   Each ``torchaudio`` package is compiled against specific version of ``torch``.
   Please refer to the following table and install the correct pair of ``torch`` and ``torchaudio``.

.. note::

   Starting ``0.10``, torchaudio has CPU-only and CUDA-enabled binary distributions,
   each of which requires a corresponding PyTorch distribution.

.. note::
   This software was compiled against an unmodified copies of FFmpeg, with the specific rpath removed so as to enable the use of system libraries. The LGPL source can be downloaded from the following locations: `n4.1.8 <https://github.com/FFmpeg/FFmpeg/releases/tag/n4.1.8>`__ (`license <https://github.com/FFmpeg/FFmpeg/blob/n4.1.8/COPYING.LGPLv2.1>`__), `n5.0.3 <https://github.com/FFmpeg/FFmpeg/releases/tag/n5.0.3>`__ (`license <https://github.com/FFmpeg/FFmpeg/blob/n5.0.3/COPYING.LGPLv2.1>`__) and `n6.0 <https://github.com/FFmpeg/FFmpeg/releases/tag/n6.0>`__ (`license <https://github.com/FFmpeg/FFmpeg/blob/n6.0/COPYING.LGPLv2.1>`__).

Dependencies
------------

* `PyTorch <https://pytorch.org>`_

  Please refer to the compatibility matrix bellow for supported PyTorch versions.

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

* `FFmpeg <https://ffmpeg.org>`_.

  Required to use :py:mod:`torchaudio.io` module.
  TorchAudio official binary distributions are compatible with FFmpeg 4.1 to 4.4.
  If you need to use FFmpeg 5, please build TorchAudio from source.

* `sentencepiece <https://pypi.org/project/sentencepiece/>`_

  Required for performing automatic speech recognition with :ref:`Emformer RNN-T<RNNT>`.

* `deep-phonemizer <https://pypi.org/project/deep-phonemizer/>`_

  Required for performing text-to-speech with :ref:`Tacotron2`.

* `kaldi_io <https://pypi.org/project/kaldi-io/>`_

  Required to use :py:mod:`torchaudio.kaldi_io` module.

   
Compatibility Matrix
--------------------

The official binary distributions of TorchAudio contain extension modules
which are written in C++ and linked against specific versions of PyTorch.

TorchAudio and PyTorch from different releases cannot be used together.
Please refer to the following table for the matching versions.

.. list-table::
   :header-rows: 1

   * - ``PyTorch``
     - ``TorchAudio``
     - ``Python``
   * - ``2.0.1``
     - ``2.0.2``
     - ``>=3.8``, ``<=3.11``
   * - ``2.0.0``
     - ``2.0.1``
     - ``>=3.8``, ``<=3.11``
   * - ``1.13.1``
     - ``0.13.1``
     - ``>=3.7``, ``<=3.10``
   * - ``1.13.0``
     - ``0.13.0``
     - ``>=3.7``, ``<=3.10``
   * - ``1.12.1``
     - ``0.12.1``
     - ``>=3.7``, ``<=3.10``
   * - ``1.12.0``
     - ``0.12.0``
     - ``>=3.7``, ``<=3.10``
   * - ``1.11.0``
     - ``0.11.0``
     - ``>=3.7``, ``<=3.9``
   * - ``1.10.0``
     - ``0.10.0``
     - ``>=3.6``, ``<=3.9``
   * - ``1.9.1``
     - ``0.9.1``
     - ``>=3.6``, ``<=3.9``
   * - ``1.8.1``
     - ``0.8.1``
     - ``>=3.6``, ``<=3.9``
   * - ``1.7.1``
     - ``0.7.2``
     - ``>=3.6``, ``<=3.9``
   * - ``1.7.0``
     - ``0.7.0``
     - ``>=3.6``, ``<=3.8``
   * - ``1.6.0``
     - ``0.6.0``
     - ``>=3.6``, ``<=3.8``
   * - ``1.5.0``
     - ``0.5.0``
     - ``>=3.5``, ``<=3.8``
   * - ``1.4.0``
     - ``0.4.0``
     - ``==2.7``, ``>=3.5``, ``<=3.8``
