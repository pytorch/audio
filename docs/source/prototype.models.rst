torchaudio.prototype.models
===========================

.. py:module:: torchaudio.prototype.models
.. currentmodule:: torchaudio.prototype.models


The ``torchaudio.prototype.models`` subpackage contains definitions of models for addressing common audio tasks.

.. note::
   For models with pre-trained parameters, please refer to :mod:`torchaudio.prototype.pipelines` module.

Model defintions are responsible for constructing computation graphs and executing them.

Some models have complex structure and variations.
For such models, factory functions are provided.

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/prototype_model_class.rst

  ConformerWav2Vec2PretrainModel
  ConvEmformer
  HiFiGANVocoder
  SQUIM_OBJECTIVE

Wav2Vec2 Factory Functions
==========================

conformer_rnnt_model
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: conformer_rnnt_model

conformer_rnnt_base
~~~~~~~~~~~~~~~~~~~

.. autofunction:: conformer_rnnt_base

conformer_wav2vec2_model
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: conformer_wav2vec2_model

conformer_wav2vec2_base
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: conformer_wav2vec2_base

emformer_hubert_model
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: emformer_hubert_model

emformer_hubert_base
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: emformer_hubert_base
