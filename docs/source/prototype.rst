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

.. autodata:: EMFORMER_RNNT_BASE_LIBRISPEECH
  :no-value:


KenLMLexiconDecoder
~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchaudio.prototype.ctc_decoder

.. autoclass:: KenLMLexiconDecoder

  .. automethod:: __call__

  .. automethod:: idxs_to_tokens


kenlm_lexicon_decoder
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchaudio.prototype.ctc_decoder

.. autoclass:: kenlm_lexicon_decoder


References
~~~~~~~~~~

.. footbibliography::
