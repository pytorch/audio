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


HiFiGAN Generator
-----------------

Interface
~~~~~~~~~

``HiFiGANGeneratorBundle`` defines HiFiGAN Generator pipeline capable of transforming Mel spectrograms into waveforms.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/bundle_class.rst

   HiFiGANGeneratorBundle

.. minigallery:: torchaudio.prototype.pipelines.HiFiGANGeneratorBundle

Pretrained Models
~~~~~~~~~~~~~~~~~

.. container:: py attribute

   .. autodata:: HIFIGAN_GENERATOR_LJSPEECH_V3
      :no-value:
