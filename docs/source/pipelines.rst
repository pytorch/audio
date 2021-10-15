torchaudio.pipelines
====================

.. currentmodule:: torchaudio.pipelines

The pipelines subpackage contains API to access the models with pretrained weights, and information/helper functions associated the pretrained weights.

wav2vec 2.0 / HuBERT - Representation Learning
----------------------------------------------

.. autoclass:: Wav2Vec2Bundle
   :members: sample_rate

   .. automethod:: get_model

WAV2VEC2_BASE
-------------

.. container:: py attribute

   .. autodata:: WAV2VEC2_BASE
      :no-value:

WAV2VEC2_LARGE
--------------

.. container:: py attribute

   .. autodata:: WAV2VEC2_LARGE
      :no-value:

WAV2VEC2_LARGE_LV60K
--------------------

.. container:: py attribute

   .. autodata:: WAV2VEC2_LARGE_LV60K
      :no-value:


WAV2VEC2_XLSR53
---------------

.. container:: py attribute

   .. autodata:: WAV2VEC2_XLSR53
      :no-value:

HUBERT_BASE
-----------

.. container:: py attribute

   .. autodata:: HUBERT_BASE
      :no-value:

HUBERT_LARGE
------------

.. container:: py attribute

   .. autodata:: HUBERT_LARGE
      :no-value:

HUBERT_XLARGE
-------------

.. container:: py attribute

   .. autodata:: HUBERT_XLARGE
      :no-value:

wav2vec 2.0 / HuBERT - ASR fine-tuning
--------------------------------------

.. autoclass:: Wav2Vec2ASRBundle
   :members: sample_rate

   .. automethod:: get_model

   .. automethod:: get_labels


WAV2VEC2_ASR_BASE_10M
---------------------

.. container:: py attribute

   .. autodata:: WAV2VEC2_ASR_BASE_10M
      :no-value:

WAV2VEC2_ASR_BASE_100H
----------------------
      
.. container:: py attribute

   .. autodata:: WAV2VEC2_ASR_BASE_100H
      :no-value:

WAV2VEC2_ASR_BASE_960H
----------------------

.. container:: py attribute

   .. autodata:: WAV2VEC2_ASR_BASE_960H
      :no-value:

WAV2VEC2_ASR_LARGE_10M
----------------------

.. container:: py attribute

   .. autodata:: WAV2VEC2_ASR_LARGE_10M
      :no-value:

WAV2VEC2_ASR_LARGE_100H
-----------------------

.. container:: py attribute

   .. autodata:: WAV2VEC2_ASR_LARGE_100H
      :no-value:

WAV2VEC2_ASR_LARGE_960H
-----------------------

.. container:: py attribute

   .. autodata:: WAV2VEC2_ASR_LARGE_960H
      :no-value:

WAV2VEC2_ASR_LARGE_LV60K_10M
----------------------------

.. container:: py attribute

   .. autodata:: WAV2VEC2_ASR_LARGE_LV60K_10M
      :no-value:

WAV2VEC2_ASR_LARGE_LV60K_100H
-----------------------------

.. container:: py attribute

   .. autodata:: WAV2VEC2_ASR_LARGE_LV60K_100H
      :no-value:

WAV2VEC2_ASR_LARGE_LV60K_960H
-----------------------------

.. container:: py attribute

   .. autodata:: WAV2VEC2_ASR_LARGE_LV60K_960H
      :no-value:

HUBERT_ASR_LARGE
----------------

.. container:: py attribute

   .. autodata:: HUBERT_ASR_LARGE
      :no-value:

HUBERT_ASR_XLARGE
-----------------

.. container:: py attribute

   .. autodata:: HUBERT_ASR_XLARGE
      :no-value:

References
----------

.. footbibliography::
