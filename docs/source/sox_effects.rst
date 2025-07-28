.. py:module:: torchaudio.sox_effects

torchaudio.sox_effects
======================

.. currentmodule:: torchaudio.sox_effects

.. warning::
    Starting with version 2.8, we are refactoring TorchAudio to transition it
    into a maintenance phase. As a result, the ``sox_effect`` module is
    deprecated in 2.8 and will be removed in 2.9.

Applying effects
----------------

Apply SoX effects chain on torch.Tensor or on file and load as torch.Tensor.

.. autosummary::
   :toctree: generated
   :nosignatures:

   apply_effects_tensor
   apply_effects_file

.. minigallery:: torchaudio.sox_effects.apply_effects_tensor
   
Utilities
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   effect_names
