
torchaudio.prototype.pipelines
==============================

.. py:module:: torchaudio.prototype.pipelines
.. currentmodule:: torchaudio.prototype.pipelines

The pipelines subpackage contains APIs to models with pretrained weights and relevant utilities.

RNN-T Streaming/Non-Streaming ASR
---------------------------------

Pretrained Models
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/bundle_data.rst

   EMFORMER_RNNT_BASE_MUSTC
   EMFORMER_RNNT_BASE_TEDLIUM3

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

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/bundle_data.rst

   HIFIGAN_VOCODER_V3_LJSPEECH

Squim Objective
---------------

Interface
~~~~~~~~~

:py:class:`SquimObjectiveBundle` defines speech quality and intelligibility measurement (SQUIM) pipeline that can predict **objecive** metric scores given the input waveform.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/bundle_class.rst

   SquimObjectiveBundle

Pretrained Models
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/bundle_data.rst

   SQUIM_OBJECTIVE
