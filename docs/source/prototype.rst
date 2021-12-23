.. role:: hidden
    :class: hidden-section

torchaudio.prototype
====================

.. py:module:: torchaudio.prototype

.. currentmodule:: torchaudio.prototype

``torchaudio.prototype`` provides prototype features;
see `here <https://pytorch.org/audio>`_ for more information on prototype features.
The module is available only within nightly builds and must be imported
explicitly, e.g. ``import torchaudio.prototype``.

Conformer
~~~~~~~~~

.. autoclass:: Conformer

  .. automethod:: forward


Emformer
~~~~~~~~

.. autoclass:: Emformer

  .. automethod:: forward

  .. automethod:: infer


RNNT
~~~~

.. autoclass:: RNNT

  .. automethod:: forward

  .. automethod:: transcribe_streaming

  .. automethod:: transcribe

  .. automethod:: predict

  .. automethod:: join

emformer_rnnt_base
~~~~~~~~~~~~~~~~~~

.. autofunction:: emformer_rnnt_base

emformer_rnnt_model
~~~~~~~~~~~~~~~~~~~

.. autofunction:: emformer_rnnt_model


RNNTBeamSearch
~~~~~~~~~~~~~~

.. autoclass:: RNNTBeamSearch

  .. automethod:: forward

  .. automethod:: infer


Hypothesis
~~~~~~~~~~

.. autoclass:: Hypothesis


References
~~~~~~~~~~

.. footbibliography::
