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
   This software was compiled against an unmodified copies of FFmpeg, with the specific rpath removed so as to enable the use of system libraries. The LGPL source can be downloaded from the following locations: `n4.1.8 <https://github.com/FFmpeg/FFmpeg/releases/tag/n4.4.4>`__ (`license <https://github.com/FFmpeg/FFmpeg/blob/n4.4.4/COPYING.LGPLv2.1>`__), `n5.0.3 <https://github.com/FFmpeg/FFmpeg/releases/tag/n5.0.3>`__ (`license <https://github.com/FFmpeg/FFmpeg/blob/n5.0.3/COPYING.LGPLv2.1>`__) and `n6.0 <https://github.com/FFmpeg/FFmpeg/releases/tag/n6.0>`__ (`license <https://github.com/FFmpeg/FFmpeg/blob/n6.0/COPYING.LGPLv2.1>`__).

Dependencies
------------

* `PyTorch <https://pytorch.org>`_

  Please refer to the compatibility matrix bellow for supported PyTorch versions.

.. _optional_dependencies:

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

.. _ffmpeg_dependency:

* `FFmpeg <https://ffmpeg.org>`__

  Required to use :py:mod:`torchaudio.io` module. and ``backend="ffmpeg"`` in
  `I/O functions <./torchaudio.html#i-o>`__.

  Starting version 2.1, TorchAudio official binary distributions are compatible with
  FFmpeg version 6, 5 and 4. (>=4.4, <7). At runtime, TorchAudio first looks for FFmpeg 6,
  if not found, then it continues to looks for 5 and move on to 4.

  There are multiple ways to install FFmpeg libraries.
  Please refer to the official documentation for how to install FFmpeg.
  If you are using Anaconda Python distribution,
  ``conda install -c conda-forge 'ffmpeg<7'`` will install
  compatible FFmpeg libraries.

  If you need to specify the version of FFmpeg TorchAudio searches and links, you can
  specify it via the environment variable ``TORCHAUDIO_USE_FFMPEG_VERSION``. For example,
  by setting ``TORCHAUDIO_USE_FFMPEG_VERSION=5``, TorchAudio will only look for FFmpeg
  5.

  If for some reason, this search mechanism is causing an issue, you can disable
  the FFmpeg integration entirely by setting the environment variable
  ``TORCHAUDIO_USE_FFMPEG=0``.

  There are multiple ways to install FFmpeg libraries.
  If you are using Anaconda Python distribution,
  ``conda install -c conda-forge 'ffmpeg<7'`` will install
  compatible FFmpeg libraries.

  .. note::

     When searching for FFmpeg installation, TorchAudio looks for library files
     which have names with version numbers.
     That is, ``libavutil.so.<VERSION>`` for Linux, ``libavutil.<VERSION>.dylib``
     for macOS, and ``avutil-<VERSION>.dll`` for Windows.
     Many public pre-built binaries follow this naming scheme, but some distributions
     have un-versioned file names.
     If you are having difficulties detecting FFmpeg, double check that the library
     files you installed follow this naming scheme, (and then make sure
     that they are in one of the directories listed in library search path.)

* `SoX <https://sox.sourceforge.net/>`__

  Required to use ``backend="sox"`` in `I/O functions <./torchaudio.html#i-o>`__.

  Starting version 2.1, TorchAudio requires separately installed libsox.

  If dynamic linking is causing an issue, you can set the environment variable
  ``TORCHAUDIO_USE_SOX=0``, and TorchAudio won't use SoX.

  .. note::

     TorchAudio looks for a library file with unversioned name, that is ``libsox.so``
     for Linux, and ``libsox.dylib`` for macOS. Some package managers install the library
     file with different name. For example, aptitude on Ubuntu installs ``libsox.so.3``.
     To have TorchAudio link against it, you can create a symbolic link to it with name
     ``libsox.so`` (and put the symlink in a library search path).

  .. note::
     TorchAudio is tested on libsox 14.4.2. (And it is unlikely that other
     versions would work.)

* `SoundFile <https://pysoundfile.readthedocs.io/>`__

  Required to use ``backend="soundfile"`` in `I/O functions <./torchaudio.html#i-o>`__.

* `sentencepiece <https://pypi.org/project/sentencepiece/>`__

  Required for performing automatic speech recognition with :ref:`Emformer RNN-T<RNNT>`.
  You can install it by running ``pip install sentencepiece``.

* `deep-phonemizer <https://pypi.org/project/deep-phonemizer/>`__

  Required for performing text-to-speech with :ref:`Tacotron2`.

* `kaldi_io <https://pypi.org/project/kaldi-io/>`__

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
