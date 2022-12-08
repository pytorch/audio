.. py:module:: torchaudio.models

torchaudio.models
=================

.. currentmodule:: torchaudio.models

The ``torchaudio.models`` subpackage contains definitions of models for addressing common audio tasks.

For pre-trained models, please refer to :mod:`torchaudio.pipelines` module.

Model Definitions
-----------------

Model defintions are responsible for constructing computation graphs and executing them.

Some models have complex structure and variations.
For such models, `Factory Functions`_ are provided.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/model_class.rst

   Conformer
   ConvTasNet
   DeepSpeech
   Emformer
   HDemucs
   HuBERTPretrainModel
   RNNT
   RNNTBeamSearch
   Tacotron2
   Wav2Letter
   Wav2Vec2Model
   WaveRNN

Factory Functions
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   conv_tasnet_base
   emformer_rnnt_model
   emformer_rnnt_base
   wav2vec2_model
   wav2vec2_base
   wav2vec2_large
   wav2vec2_large_lv60k
   hubert_base
   hubert_large
   hubert_xlarge
   hubert_pretrain_model
   hubert_pretrain_base
   hubert_pretrain_large
   hubert_pretrain_xlarge
   hdemucs_low
   hdemucs_medium
   hdemucs_high
   wavlm_model
   wavlm_base
   wavlm_large

Utility Functions
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~wav2vec2.utils.import_fairseq_model
   ~wav2vec2.utils.import_huggingface_model
