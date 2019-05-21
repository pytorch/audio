torchaudio.kaldi_io
======================

.. currentmodule:: torchaudio.kaldi_io

To use this module, the dependency kaldi_io_ needs to be installed.
This is a light wrapper around ``kaldi_io`` that returns :class:`torch.Tensors`.

.. _kaldi_io: https://github.com/vesis84/kaldi-io-for-python

Vectors
~~~~~

.. autodata:: read_vec_int_ark

.. autodata:: read_vec_flt_scp

.. autodata:: read_vec_flt_ark

Matrices
~~~~~

.. autodata:: read_mat_scp

.. autodata:: read_mat_ark
