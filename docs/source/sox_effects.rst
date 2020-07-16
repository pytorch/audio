.. role:: hidden
    :class: hidden-section

torchaudio.sox_effects
======================

.. currentmodule:: torchaudio.sox_effects

Apply SoX effects chain on torch.Tensor or on file and load as torch.Tensor.

.. autofunction:: apply_effects_tensor

.. autofunction:: apply_effects_file

Create SoX effects chain for preprocessing audio.

:hidden:`SoxEffect`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SoxEffect
  :members:

:hidden:`SoxEffectsChain`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SoxEffectsChain
  :members: append_effect_to_chain, sox_build_flow_effects, clear_chain, set_input_file
