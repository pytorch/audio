.. py:module:: torchaudio.pipelines

torchaudio.pipelines
====================

.. currentmodule:: torchaudio.pipelines
		   
The ``torchaudio.pipelines`` module packages pre-trained models with support functions and meta-data into simple APIs tailored to perform specific tasks.

When using pre-trained models to perform a task, in addition to instantiating the model with pre-trained weights, the client code also needs to build pipelines for feature extractions and post processing in the same way they were done during the training. This requires to carrying over information used during the training, such as the type of transforms and the their parameters (for example, sampling rate the number of FFT bins).

To make this information tied to a pre-trained model and easily accessible, ``torchaudio.pipelines`` module uses the concept of a `Bundle` class, which defines a set of APIs to instantiate pipelines, and the interface of the pipelines.

The following figure illustrates this.

.. image:: https://download.pytorch.org/torchaudio/doc-assets/pipelines-intro.png

A pre-trained model and associated pipelines are expressed as an instance of ``Bundle``. Different instances of same ``Bundle`` share the interface, but their implementations are not constrained to be of same types. For example, :class:`SourceSeparationBundle` defines the interface for performing source separation, but its instance :data:`CONVTASNET_BASE_LIBRI2MIX` instantiates a model of :class:`~torchaudio.models.ConvTasNet` while :data:`HDEMUCS_HIGH_MUSDB` instantiates a model of :class:`~torchaudio.models.HDemucs`. Still, because they share the same interface, the usage is the same.

.. note::

   Under the hood, the implementations of ``Bundle`` use components from other ``torchaudio`` modules, such as :mod:`torchaudio.models` and :mod:`torchaudio.transforms`, or even third party libraries like `SentencPiece <https://github.com/google/sentencepiece>`__ and `DeepPhonemizer <https://github.com/as-ideas/DeepPhonemizer>`__. But this implementation detail is abstracted away from library users.

.. _RNNT:

RNN-T Streaming/Non-Streaming ASR
---------------------------------

Interface
~~~~~~~~~

``RNNTBundle`` defines ASR pipelines and consists of three steps: feature extraction, inference, and de-tokenization.

.. image:: https://download.pytorch.org/torchaudio/doc-assets/pipelines-rnntbundle.png

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/bundle_class.rst

   RNNTBundle
   RNNTBundle.FeatureExtractor
   RNNTBundle.TokenProcessor

.. rubric:: Tutorials using ``RNNTBundle``

.. minigallery:: torchaudio.pipelines.RNNTBundle

Pretrained Models
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/bundle_data.rst

   EMFORMER_RNNT_BASE_LIBRISPEECH


wav2vec 2.0 / HuBERT / WavLM - SSL
----------------------------------

Interface
~~~~~~~~~

``Wav2Vec2Bundle`` instantiates models that generate acoustic features that can be used for downstream inference and fine-tuning.

.. image:: https://download.pytorch.org/torchaudio/doc-assets/pipelines-wav2vec2bundle.png

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/bundle_class.rst

   Wav2Vec2Bundle

Pretrained Models
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/bundle_data.rst

   WAV2VEC2_BASE
   WAV2VEC2_LARGE
   WAV2VEC2_LARGE_LV60K
   WAV2VEC2_XLSR53
   WAV2VEC2_XLSR_300M
   WAV2VEC2_XLSR_1B
   WAV2VEC2_XLSR_2B
   HUBERT_BASE
   HUBERT_LARGE
   HUBERT_XLARGE
   WAVLM_BASE
   WAVLM_BASE_PLUS
   WAVLM_LARGE

wav2vec 2.0 / HuBERT - Fine-tuned ASR
-------------------------------------

Interface
~~~~~~~~~

``Wav2Vec2ASRBundle`` instantiates models that generate probability distribution over pre-defined labels, that can be used for ASR.

.. image:: https://download.pytorch.org/torchaudio/doc-assets/pipelines-wav2vec2asrbundle.png

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/bundle_class.rst

   Wav2Vec2ASRBundle

.. rubric:: Tutorials using ``Wav2Vec2ASRBundle``

.. minigallery:: torchaudio.pipelines.Wav2Vec2ASRBundle

Pretrained Models
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/bundle_data.rst

   WAV2VEC2_ASR_BASE_10M
   WAV2VEC2_ASR_BASE_100H
   WAV2VEC2_ASR_BASE_960H
   WAV2VEC2_ASR_LARGE_10M
   WAV2VEC2_ASR_LARGE_100H
   WAV2VEC2_ASR_LARGE_960H
   WAV2VEC2_ASR_LARGE_LV60K_10M
   WAV2VEC2_ASR_LARGE_LV60K_100H
   WAV2VEC2_ASR_LARGE_LV60K_960H
   VOXPOPULI_ASR_BASE_10K_DE
   VOXPOPULI_ASR_BASE_10K_EN
   VOXPOPULI_ASR_BASE_10K_ES
   VOXPOPULI_ASR_BASE_10K_FR
   VOXPOPULI_ASR_BASE_10K_IT
   HUBERT_ASR_LARGE
   HUBERT_ASR_XLARGE

wav2vec 2.0 / HuBERT - Forced Alignment
---------------------------------------

Interface
~~~~~~~~~

``Wav2Vec2FABundle`` bundles pre-trained model and its associated dictionary. Additionally, it supports appending ``star`` token dimension.

.. image:: https://download.pytorch.org/torchaudio/doc-assets/pipelines-wav2vec2fabundle.png

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/bundle_class.rst

   Wav2Vec2FABundle
   Wav2Vec2FABundle.Tokenizer
   Wav2Vec2FABundle.Aligner

.. rubric:: Tutorials using ``Wav2Vec2FABundle``

.. minigallery:: torchaudio.pipelines.Wav2Vec2FABundle

Pertrained Models
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/bundle_data.rst

   MMS_FA

.. _Tacotron2:
   
Tacotron2 Text-To-Speech
------------------------

``Tacotron2TTSBundle`` defines text-to-speech pipelines and consists of three steps: tokenization, spectrogram generation and vocoder. The spectrogram generation is based on :class:`~torchaudio.models.Tacotron2` model.

.. image:: https://download.pytorch.org/torchaudio/doc-assets/pipelines-tacotron2bundle.png

``TextProcessor`` can be rule-based tokenization in the case of characters, or it can be a neural-netowrk-based G2P model that generates sequence of phonemes from input text.

Similarly ``Vocoder`` can be an algorithm without learning parameters, like `Griffin-Lim`, or a neural-network-based model like `Waveglow`.

Interface
~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/bundle_class.rst

   Tacotron2TTSBundle
   Tacotron2TTSBundle.TextProcessor
   Tacotron2TTSBundle.Vocoder

.. rubric:: Tutorials using ``Tacotron2TTSBundle``

.. minigallery:: torchaudio.pipelines.Tacotron2TTSBundle

Pretrained Models
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/bundle_data.rst

   TACOTRON2_WAVERNN_PHONE_LJSPEECH
   TACOTRON2_WAVERNN_CHAR_LJSPEECH
   TACOTRON2_GRIFFINLIM_PHONE_LJSPEECH
   TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH

Source Separation
-----------------

Interface
~~~~~~~~~

``SourceSeparationBundle`` instantiates source separation models which take single channel audio and generates multi-channel audio.

.. image:: https://download.pytorch.org/torchaudio/doc-assets/pipelines-sourceseparationbundle.png

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/bundle_class.rst

   SourceSeparationBundle

.. rubric:: Tutorials using ``SourceSeparationBundle``

.. minigallery:: torchaudio.pipelines.SourceSeparationBundle

Pretrained Models
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/bundle_data.rst

   CONVTASNET_BASE_LIBRI2MIX
   HDEMUCS_HIGH_MUSDB_PLUS
   HDEMUCS_HIGH_MUSDB

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

Squim Subjective
----------------

Interface
~~~~~~~~~

:py:class:`SquimSubjectiveBundle` defines speech quality and intelligibility measurement (SQUIM) pipeline that can predict **subjective** metric scores given the input waveform.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/bundle_class.rst

   SquimSubjectiveBundle

Pretrained Models
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/bundle_data.rst

   SQUIM_SUBJECTIVE
