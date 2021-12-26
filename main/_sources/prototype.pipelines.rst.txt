torchaudio.prototype.pipelines
==============================

.. py:module:: torchaudio.prototype.pipelines

.. currentmodule:: torchaudio.prototype.pipelines

The pipelines subpackage contains APIs to models with pretrained weights and relevant utilities.

RNN-T Streaming/Non-Streaming ASR
---------------------------------

RNNTBundle
~~~~~~~~~~

.. autoclass:: RNNTBundle
  :members: sample_rate, n_fft, n_mels, hop_length, segment_length, right_context_length

  .. automethod:: get_decoder() -> torchaudio.prototype.models.RNNTBeamSearch

  .. automethod:: get_feature_extractor() -> RNNTBundle.FeatureExtractor

  .. automethod:: get_streaming_feature_extractor() -> RNNTBundle.FeatureExtractor

  .. automethod:: get_token_processor() -> RNNTBundle.TokenProcessor

RNNTBundle - FeatureExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchaudio.prototype.pipelines::RNNTBundle.FeatureExtractor
  :special-members: __call__

RNNTBundle - TokenProcessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchaudio.prototype.pipelines::RNNTBundle.TokenProcessor
  :special-members: __call__

EMFORMER_RNNT_BASE_LIBRISPEECH
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container:: py attribute

   .. autodata:: EMFORMER_RNNT_BASE_LIBRISPEECH
      :no-value:
