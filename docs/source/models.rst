.. role:: hidden
    :class: hidden-section

torchaudio.models
=================

.. currentmodule:: torchaudio.models

The models subpackage contains definitions of models for addressing common audio tasks.


ConvTasNet
~~~~~~~~~~

.. autoclass:: ConvTasNet

  .. automethod:: forward


DeepSpeech
~~~~~~~~~~

.. autoclass:: DeepSpeech

  .. automethod:: forward


Tacotron2
~~~~~~~~~

Model
-----

Tacotron2
^^^^^^^^^

.. autoclass:: Tacotron2

  .. automethod:: forward

  .. automethod:: infer

Factory Functions
-----------------

tacotron2
^^^^^^^^^

.. autofunction:: tacotron2


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

Model
-----

WaveRNN
^^^^^^^

.. autoclass:: WaveRNN

  .. automethod:: forward

  .. automethod:: infer

Factory Functions
-----------------

wavernn
^^^^^^^

.. autofunction:: wavernn

References
~~~~~~~~~~

.. footbibliography::
