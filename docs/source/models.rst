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

Tacotoron2
^^^^^^^^^^

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

wav2vec2_ft_base
^^^^^^^^^^^^^^^^^

.. autofunction:: wav2vec2_ft_base

wav2vec2_ft_large
^^^^^^^^^^^^^^^^^^

.. autofunction:: wav2vec2_ft_large

wav2vec2_ft_large_lv60k
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: wav2vec2_ft_large_lv60k

hubert_base
^^^^^^^^^^^

.. autofunction:: hubert_base

hubert_large
^^^^^^^^^^^^

.. autofunction:: hubert_large

hubert_xlarge
^^^^^^^^^^^^^

.. autofunction:: hubert_xlarge

hubert_ft_large
^^^^^^^^^^^^^^^^

.. autofunction:: hubert_ft_large

hubert_ft_xlarge
^^^^^^^^^^^^^^^^^

.. autofunction:: hubert_ft_xlarge

Pre-trained Models
------------------

.. autoclass:: Wav2Vec2PretrainedModelBundle

   .. automethod:: get_model

   .. autoproperty:: labels


WAV2VEC2_BASE
^^^^^^^^^^^^^

.. container:: py attribute

   .. autodata:: WAV2VEC2_BASE
      :no-value:

WAV2VEC2_ASR_BASE_10M
^^^^^^^^^^^^^^^^^^^^^

.. container:: py attribute

   .. autodata:: torchaudio.models.WAV2VEC2_ASR_BASE_10M
      :no-value:

WAV2VEC2_ASR_BASE_100H
^^^^^^^^^^^^^^^^^^^^^^
      
.. container:: py attribute

   .. autodata:: WAV2VEC2_ASR_BASE_100H
      :no-value:

WAV2VEC2_ASR_BASE_960H
^^^^^^^^^^^^^^^^^^^^^^

.. container:: py attribute

   .. autodata:: WAV2VEC2_ASR_BASE_960H
      :no-value:

WAV2VEC2_LARGE
^^^^^^^^^^^^^^

.. container:: py attribute

   .. autodata:: WAV2VEC2_LARGE
      :no-value:

WAV2VEC2_ASR_LARGE_10M
^^^^^^^^^^^^^^^^^^^^^^

.. container:: py attribute

   .. autodata:: WAV2VEC2_ASR_LARGE_10M
      :no-value:

WAV2VEC2_ASR_LARGE_100H
^^^^^^^^^^^^^^^^^^^^^^^

.. container:: py attribute

   .. autodata:: WAV2VEC2_ASR_LARGE_100H
      :no-value:

WAV2VEC2_ASR_LARGE_960H
^^^^^^^^^^^^^^^^^^^^^^^

.. container:: py attribute

   .. autodata:: WAV2VEC2_ASR_LARGE_960H
      :no-value:

WAV2VEC2_LARGE_LV60K
^^^^^^^^^^^^^^^^^^^^

.. container:: py attribute

   .. autodata:: WAV2VEC2_LARGE_LV60K
      :no-value:

WAV2VEC2_ASR_LARGE_LV60K_10M
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. container:: py attribute

   .. autodata:: WAV2VEC2_ASR_LARGE_LV60K_10M
      :no-value:

WAV2VEC2_ASR_LARGE_LV60K_100H
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. container:: py attribute

   .. autodata:: WAV2VEC2_ASR_LARGE_LV60K_100H
      :no-value:

WAV2VEC2_ASR_LARGE_LV60K_960H
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. container:: py attribute

   .. autodata:: WAV2VEC2_ASR_LARGE_LV60K_960H
      :no-value:

WAV2VEC2_XLSR53
^^^^^^^^^^^^^^^

.. container:: py attribute

   .. autodata:: WAV2VEC2_XLSR53
      :no-value:

HUBERT_BASE
^^^^^^^^^^^

.. container:: py attribute

   .. autodata:: HUBERT_BASE
      :no-value:

HUBERT_LARGE
^^^^^^^^^^^^

.. container:: py attribute

   .. autodata:: HUBERT_LARGE
      :no-value:

HUBERT_XLARGE
^^^^^^^^^^^^^

.. container:: py attribute

   .. autodata:: HUBERT_XLARGE
      :no-value:

HUBERT_ASR_LARGE
^^^^^^^^^^^^^^^^

.. container:: py attribute

   .. autodata:: HUBERT_ASR_LARGE
      :no-value:

HUBERT_ASR_XLARGE
^^^^^^^^^^^^^^^^^

.. container:: py attribute

   .. autodata:: HUBERT_ASR_XLARGE
      :no-value:

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
