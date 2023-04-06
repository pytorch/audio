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
  SquimObjective
  SquimSubjective

Prototype Factory Functions of Beta Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchaudio.models

Some model definitions are in beta, but there are new factory functions that are still in prototype. Please check "Prototype Factory Functions" section in each model.

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/model_class.rst

  Wav2Vec2Model
  RNNT
