.. role:: hidden
    :class: hidden-section

torchaudio.kaldi_io
======================

.. py:module:: torchaudio.kaldi_io

.. currentmodule:: torchaudio.kaldi_io

.. warning::
    Starting with version 2.8, we are refactoring TorchAudio to transition it
    into a maintenance phase. As a result, the ``kaldi_io`` module is
    deprecated in 2.8 and will be removed in 2.9.

To use this module, the dependency kaldi_io_ needs to be installed.
This is a light wrapper around ``kaldi_io`` that returns :class:`torch.Tensor`.

.. _kaldi_io: https://github.com/vesis84/kaldi-io-for-python

Vectors
-------

:hidden:`read_vec_int_ark`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: read_vec_int_ark

:hidden:`read_vec_flt_scp`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: read_vec_flt_scp

:hidden:`read_vec_flt_ark`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: read_vec_flt_ark

Matrices
--------

:hidden:`read_mat_scp`
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: read_mat_scp

:hidden:`read_mat_ark`
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: read_mat_ark
