torchaudio.prototype.models
===========================

.. py:module:: torchaudio.prototype.models

.. currentmodule:: torchaudio.prototype.models

The models subpackage contains definitions of models and components for addressing common audio tasks.


Model Classes
-------------

Conformer
~~~~~~~~~

.. autoclass:: Conformer

  .. automethod:: forward

Emformer
~~~~~~~~

.. autoclass:: Emformer

  .. automethod:: forward

  .. automethod:: infer

RNNT
~~~~

.. autoclass:: RNNT

  .. automethod:: forward

  .. automethod:: transcribe_streaming

  .. automethod:: transcribe

  .. automethod:: predict

  .. automethod:: join


Model Factory Functions
-----------------------

emformer_rnnt_model
~~~~~~~~~~~~~~~~~~~

.. autofunction:: emformer_rnnt_model

emformer_rnnt_base
~~~~~~~~~~~~~~~~~~

.. autofunction:: emformer_rnnt_base


Decoder Classes
---------------

RNNTBeamSearch
~~~~~~~~~~~~~~

.. autoclass:: RNNTBeamSearch

  .. automethod:: forward

  .. automethod:: infer

Hypothesis
~~~~~~~~~~

.. autoclass:: Hypothesis


References
----------

.. footbibliography::
