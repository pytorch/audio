.. _sox_effects:

torchaudio.sox_effects
======================

.. currentmodule:: torchaudio.sox_effects

.. warning::

   The :py:class:`SoxEffect` and :py:class:`SoxEffectsChain` classes are deprecated. Please migrate to :func:`apply_effects_tensor` and :func:`apply_effects_file`.

Resource initialization / shutdown
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: init_sox_effects

.. autofunction:: shutdown_sox_effects

Listing supported effects
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: effect_names

Applying effects
~~~~~~~~~~~~~~~~

Apply SoX effects chain on torch.Tensor or on file and load as torch.Tensor.

Applying effects on Tensor
--------------------------

.. autofunction:: apply_effects_tensor

Applying effects on file
------------------------

.. autofunction:: apply_effects_file

Legacy
~~~~~~

SoxEffect
---------

.. autoclass:: SoxEffect
  :members:

SoxEffectsChain
---------------

.. autoclass:: SoxEffectsChain
  :members: append_effect_to_chain, sox_build_flow_effects, clear_chain, set_input_file
