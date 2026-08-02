Installing pre-built binaries
=============================

``torchaudio`` has binary distributions on PyPI:

.. code-block::

   pip install torchaudio

To install a specific variant (CUDA, ROCm, nightly, ...), please refer to
https://pytorch.org/get-started/locally/.

.. important::

   **TorchAudio 2.11 works with PyTorch 2.11 and with every future PyTorch
   release (2.12, 2.13, ...).**

Dependencies
------------

* `PyTorch <https://pytorch.org>`_

  2.11 or newer for TorchAudio 2.11. For older TorchAudio releases, please
  refer to the :ref:`compatibility matrix <compatibility_matrix>` below.

.. _optional_dependencies:

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

* `torchcodec <https://github.com/pytorch/torchcodec>`__

  Required by :func:`torchaudio.load` and :func:`torchaudio.save`, which are
  thin wrappers around TorchCodec's ``AudioDecoder`` and ``AudioEncoder``.
  We recommend using those TorchCodec classes directly. Installation
  instructions are at
  https://github.com/pytorch/torchcodec#installing-torchcodec.

* `sentencepiece <https://pypi.org/project/sentencepiece/>`__

  Required for performing automatic speech recognition with :ref:`Emformer RNN-T<RNNT>`.
  You can install it by running ``pip install sentencepiece``.

* `deep-phonemizer <https://pypi.org/project/deep-phonemizer/>`__

  Required for performing text-to-speech with :ref:`Tacotron2`.

.. _compatibility_matrix:

Compatibility Matrix
--------------------

TorchAudio 2.11 is built against PyTorch's stable ABI and therefore supports
PyTorch 2.11 and all later versions. Earlier TorchAudio releases contain
extension modules linked against a single PyTorch version, and cannot be mixed
with a different PyTorch release.

.. list-table::
   :header-rows: 1

   * - ``PyTorch``
     - ``TorchAudio``
     - ``Python``
   * - ``2.11`` **and above**
     - ``2.11.0``
     - ``>=3.10``, ``<=3.14``
   * - ``2.10``
     - ``2.10.0``
     - ``>=3.10``, ``<=3.14``
   * - ``2.9.1``
     - ``2.9.1``
     - ``>=3.10``, ``<=3.14``
   * - ``2.9``
     - ``2.9.0``
     - ``>=3.10``, ``<=3.14``
   * - ``2.8``
     - ``2.8.0``
     - ``>=3.9``, ``<=3.13``
   * - ``2.7.1``
     - ``2.7.1``
     - ``>=3.9``, ``<=3.13``
   * - ``2.7``
     - ``2.7.0``
     - ``>=3.9``, ``<=3.13``
   * - ``2.6``
     - ``2.6.0``
     - ``>=3.9``, ``<=3.13``
   * - ``2.5``
     - ``2.5.0``
     - ``>=3.8``, ``<=3.11``
   * - ``2.4.1``
     - ``2.4.1``
     - ``>=3.8``, ``<=3.11``
   * - ``2.4``
     - ``2.4.0``
     - ``>=3.8``, ``<=3.11``
   * - ``2.3.1``
     - ``2.3.1``
     - ``>=3.8``, ``<=3.11``
   * - ``2.3.0``
     - ``2.3.0``
     - ``>=3.8``, ``<=3.11``
   * - ``2.2.2``
     - ``2.2.2``
     - ``>=3.8``, ``<=3.11``
   * - ``2.2.1``
     - ``2.2.1``
     - ``>=3.8``, ``<=3.11``
   * - ``2.2``
     - ``2.2.0``
     - ``>=3.8``, ``<=3.11``
   * - ``2.1.2``
     - ``2.1.2``
     - ``>=3.8``, ``<=3.11``
   * - ``2.1.1``
     - ``2.1.1``
     - ``>=3.8``, ``<=3.11``
   * - ``2.1.0``
     - ``2.1.0``
     - ``>=3.8``, ``<=3.11``
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
