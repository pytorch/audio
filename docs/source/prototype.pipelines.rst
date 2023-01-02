torchaudio.prototype.pipelines
==============================

.. py:module:: torchaudio.prototype.pipelines
.. currentmodule:: torchaudio.prototype.pipelines

The pipelines subpackage contains APIs to models with pretrained weights and relevant utilities.

RNN-T Streaming/Non-Streaming ASR
---------------------------------

EMFORMER_RNNT_BASE_MUSTC
~~~~~~~~~~~~~~~~~~~~~~~~

.. container:: py attribute

   .. autodata:: EMFORMER_RNNT_BASE_MUSTC
      :no-value:

EMFORMER_RNNT_BASE_TEDLIUM3
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container:: py attribute

   .. autodata:: EMFORMER_RNNT_BASE_TEDLIUM3
      :no-value:


HiFiGAN Vocoder
---------------

Interface
~~~~~~~~~

:py:class:`HiFiGANVocoderBundle` defines HiFiGAN Vocoder pipeline capable of transforming mel spectrograms into waveforms.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/bundle_class.rst

   HiFiGANVocoderBundle

Pretrained Models
~~~~~~~~~~~~~~~~~

.. container:: py attribute

   .. autodata:: HIFIGAN_VOCODER_V3_LJSPEECH
      :no-value:
