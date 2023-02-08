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
   This software was compiled against an unmodified copy of FFmpeg (licensed under `the LGPLv2.1 <https://github.com/FFmpeg/FFmpeg/blob/a5d2008e2a2360d351798e9abe883d603e231442/COPYING.LGPLv2.1>`_), with the specific rpath removed so as to enable the use of system libraries. The LGPL source can be downloaded `here <https://github.com/FFmpeg/FFmpeg/releases/tag/n4.1.8>`_.

Compatibility Matrix
--------------------

.. list-table::
   :header-rows: 1

   * - ``torch``
     - ``torchaudio``
     - ``python``
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
