torchaudio.prototype.rnnt
=========================

.. py:module:: torchaudio.prototype

.. currentmodule:: torchaudio.prototype

Model Classes
-------------

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

Model Factory Functions
-----------------------

emformer_rnnt_base
~~~~~~~~~~~~~~~~~~

.. autofunction:: emformer_rnnt_base

emformer_rnnt_model
~~~~~~~~~~~~~~~~~~~

.. autofunction:: emformer_rnnt_model

Decoder Classes
---------------

RNNTBeamSearch
~~~~~~~~~~~~

.. autoclass:: RNNTBeamSearch

  .. automethod:: forward

  .. automethod:: infer


Hypothesis
~~~~~~~~~~

.. autoclass:: Hypothesis

Pipeline Primitives (Pre-trained Models)
----------------------------------------

RNNTBundle
~~~~~~~~~~

.. autoclass:: RNNTBundle
  :members: sample_rate, n_fft, n_mels, hop_length, segment_length, right_context_length

  .. automethod:: get_decoder

  .. automethod:: get_feature_extractor

  .. automethod:: get_streaming_feature_extractor

  .. automethod:: get_token_processor

  .. autoclass:: torchaudio.prototype::RNNTBundle.FeatureExtractor
    :special-members: __call__

  .. autoclass:: torchaudio.prototype::RNNTBundle.TokenProcessor
    :special-members: __call__


EMFORMER_RNNT_BASE_LIBRISPEECH
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container:: py attribute

   .. autodata:: EMFORMER_RNNT_BASE_LIBRISPEECH
      :no-value:

References
----------

.. footbibliography::
