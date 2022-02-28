.. role:: hidden
    :class: hidden-section

torchaudio.models
=================

.. py:module:: torchaudio.models

.. currentmodule:: torchaudio.models

The models subpackage contains definitions of models for addressing common audio tasks.

Conformer
~~~~~~~~~

.. autoclass:: Conformer

  .. automethod:: forward

ConvTasNet
~~~~~~~~~~

.. autoclass:: ConvTasNet

  .. automethod:: forward


DeepSpeech
~~~~~~~~~~

.. autoclass:: DeepSpeech

  .. automethod:: forward

Emformer
~~~~~~~~

.. autoclass:: Emformer

  .. automethod:: forward

  .. automethod:: infer

RNN-T
~~~~~

Model
-----

RNNT
^^^^

.. autoclass:: RNNT

  .. automethod:: forward

  .. automethod:: transcribe_streaming

  .. automethod:: transcribe

  .. automethod:: predict

  .. automethod:: join

Factory Functions
-----------------

emformer_rnnt_model
^^^^^^^^^^^^^^^^^^^

.. autofunction:: emformer_rnnt_model

emformer_rnnt_base
^^^^^^^^^^^^^^^^^^

.. autofunction:: emformer_rnnt_base


Decoder
-------

RNNTBeamSearch
^^^^^^^^^^^^^^

.. autoclass:: RNNTBeamSearch

  .. automethod:: forward

  .. automethod:: infer

Hypothesis
^^^^^^^^^^

.. autoclass:: Hypothesis


Tacotron2
~~~~~~~~~

.. autoclass:: Tacotron2

  .. automethod:: forward

  .. automethod:: infer

Wav2Letter
~~~~~~~~~~

.. autoclass:: Wav2Letter

  .. automethod:: forward


Wav2Vec2.0 / HuBERT
~~~~~~~~~~~~~~~~~~~

Model
-----

Wav2Vec2Model
^^^^^^^^^^^^^

.. autoclass:: Wav2Vec2Model

  .. automethod:: extract_features

  .. automethod:: forward

HuBERTPretrainModel
^^^^^^^^^^^^^^^^^^^

.. autoclass:: HuBERTPretrainModel

  .. automethod:: forward

Factory Functions
-----------------

wav2vec2_model
^^^^^^^^^^^^^^

.. autofunction:: wav2vec2_model


wav2vec2_base
^^^^^^^^^^^^^

.. autofunction:: wav2vec2_base

wav2vec2_large
^^^^^^^^^^^^^^

.. autofunction:: wav2vec2_large

wav2vec2_large_lv60k
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: wav2vec2_large_lv60k

hubert_base
^^^^^^^^^^^

.. autofunction:: hubert_base

hubert_large
^^^^^^^^^^^^

.. autofunction:: hubert_large

hubert_xlarge
^^^^^^^^^^^^^

.. autofunction:: hubert_xlarge

hubert_pretrain_model
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: hubert_pretrain_model

hubert_pretrain_base
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: hubert_pretrain_base

hubert_pretrain_large
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: hubert_pretrain_large

hubert_pretrain_xlarge
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: hubert_pretrain_xlarge

Utility Functions
-----------------

.. currentmodule:: torchaudio.models.wav2vec2.utils

import_huggingface_model
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: import_huggingface_model

import_fairseq_model
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: import_fairseq_model

.. currentmodule:: torchaudio.models

WaveRNN
~~~~~~~

.. autoclass:: WaveRNN

  .. automethod:: forward

  .. automethod:: infer

References
~~~~~~~~~~

.. footbibliography::
