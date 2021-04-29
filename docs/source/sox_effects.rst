.. _sox_effects:

torchaudio.sox_effects
======================

.. currentmodule:: torchaudio.sox_effects

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
