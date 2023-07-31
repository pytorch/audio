.. py:module:: torchaudio.models

torchaudio.models
=================

.. currentmodule:: torchaudio.models

The ``torchaudio.models`` subpackage contains definitions of models for addressing common audio tasks.

.. note::
   For models with pre-trained parameters, please refer to :mod:`torchaudio.pipelines` module.

Model defintions are responsible for constructing computation graphs and executing them.

Some models have complex structure and variations.
For such models, factory functions are provided.

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
   SquimObjective
   SquimSubjective
   Tacotron2
   Wav2Letter
   Wav2Vec2Model
   WaveRNN
