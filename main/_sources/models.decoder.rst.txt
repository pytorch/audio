.. role:: hidden
    :class: hidden-section

torchaudio.models.decoder
=========================

.. currentmodule:: torchaudio.models.decoder

.. py:module:: torchaudio.models.decoder

Decoder Class
-------------

CTCDecoder
~~~~~~~~~~


.. autoclass:: CTCDecoder

  .. automethod:: __call__

  .. automethod:: idxs_to_tokens

CTCDecoderLM
~~~~~~~~~~~~

.. autoclass:: CTCDecoderLM

   .. automethod:: start

   .. automethod:: score

   .. automethod:: finish

CTCDecoderLMState
~~~~~~~~~~~~~~~~~

.. autoclass:: CTCDecoderLMState
   :members: children

   .. automethod:: child

   .. automethod:: compare



CTCHypothesis
~~~~~~~~~~~~~

.. autoclass:: CTCHypothesis

Factory Function
----------------

ctc_decoder
~~~~~~~~~~~

.. autoclass:: ctc_decoder

Utility Function
----------------

download_pretrained_files
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: download_pretrained_files
