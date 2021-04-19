.. role:: hidden
    :class: hidden-section

torchaudio.models
======================

.. currentmodule:: torchaudio.models

The models subpackage contains definitions of models for addressing common audio tasks.


:hidden:`ConvTasNet`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ConvTasNet

  .. automethod:: forward


:hidden:`DeepSpeech`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DeepSpeech

  .. automethod:: forward


:hidden:`Wav2Letter`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Wav2Letter

  .. automethod:: forward



:hidden:`Wav2Vec2.0`
~~~~~~~~~~~~~~~~~~~~

Model
-----

.. autoclass:: Wav2Vec2Model

  .. automethod:: extract_features

  .. automethod:: forward

Factory Functions
-----------------

.. autofunction:: wav2vec2_base

.. autofunction:: wav2vec2_large

.. autofunction:: wav2vec2_large_lv60k


:hidden:`WaveRNN`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: WaveRNN

  .. automethod:: forward
