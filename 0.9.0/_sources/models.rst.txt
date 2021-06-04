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


Wav2Letter
~~~~~~~~~~

.. autoclass:: Wav2Letter

  .. automethod:: forward



Wav2Vec2.0
~~~~~~~~~~

Wav2Vec2Model
-------------

.. autoclass:: Wav2Vec2Model

  .. automethod:: extract_features

  .. automethod:: forward

Factory Functions
-----------------

wav2vec2_base
-------------

.. autofunction:: wav2vec2_base

wav2vec2_large
--------------

.. autofunction:: wav2vec2_large

wav2vec2_large_lv60k
--------------------

.. autofunction:: wav2vec2_large_lv60k

.. currentmodule:: torchaudio.models.wav2vec2.utils

Utility Functions
-----------------

import_huggingface_model
------------------------
		   
.. autofunction:: import_huggingface_model

import_fairseq_model
--------------------
		   
.. autofunction:: import_fairseq_model

.. currentmodule:: torchaudio.models

WaveRNN
~~~~~~~

.. autoclass:: WaveRNN

  .. automethod:: forward

References
~~~~~~~~~~

.. footbibliography::

