.. role:: hidden
    :class: hidden-section

torchaudio.transforms
======================

.. py:module:: torchaudio.transforms

.. currentmodule:: torchaudio.transforms

Transforms are common audio transforms. They can be chained together using :class:`torch.nn.Sequential`

:hidden:`Utility`
~~~~~~~~~~~~~~~~~~

:hidden:`AmplitudeToDB`
-----------------------

.. autoclass:: AmplitudeToDB

  .. automethod:: forward

:hidden:`MelScale`
------------------

.. autoclass:: MelScale

  .. automethod:: forward

:hidden:`InverseMelScale`
-------------------------

.. autoclass:: InverseMelScale

  .. automethod:: forward

:hidden:`MuLawEncoding`
-----------------------

.. autoclass:: MuLawEncoding

  .. automethod:: forward

:hidden:`MuLawDecoding`
-----------------------

.. autoclass:: MuLawDecoding

  .. automethod:: forward

:hidden:`Resample`
------------------

.. autoclass:: Resample

  .. automethod:: forward

:hidden:`FrequencyMasking`
--------------------------

.. autoclass:: FrequencyMasking

  .. automethod:: forward

:hidden:`TimeMasking`
---------------------

.. autoclass:: TimeMasking

  .. automethod:: forward

:hidden:`TimeStretch`
---------------------

.. autoclass:: TimeStretch

  .. automethod:: forward

:hidden:`Fade`
--------------

.. autoclass:: Fade

  .. automethod:: forward

:hidden:`Vol`
-------------

.. autoclass:: Vol

  .. automethod:: forward

:hidden:`Feature Extractions`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:hidden:`Spectrogram`
---------------------

.. autoclass:: Spectrogram

  .. automethod:: forward

:hidden:`InverseSpectrogram`
----------------------------

.. autoclass:: InverseSpectrogram

  .. automethod:: forward

:hidden:`MelSpectrogram`
------------------------

.. autoclass:: MelSpectrogram

  .. automethod:: forward

:hidden:`GriffinLim`
--------------------

.. autoclass:: GriffinLim

  .. automethod:: forward

:hidden:`MFCC`
--------------

.. autoclass:: MFCC

  .. automethod:: forward

:hidden:`LFCC`
--------------

.. autoclass:: LFCC

  .. automethod:: forward

:hidden:`ComputeDeltas`
-----------------------

.. autoclass:: ComputeDeltas

  .. automethod:: forward

:hidden:`PitchShift`
--------------------

.. autoclass:: PitchShift

  .. automethod:: forward

:hidden:`SlidingWindowCmn`
--------------------------

.. autoclass:: SlidingWindowCmn

  .. automethod:: forward

:hidden:`SpectralCentroid`
--------------------------

.. autoclass:: SpectralCentroid

  .. automethod:: forward

:hidden:`Vad`
-------------

.. autoclass:: Vad

  .. automethod:: forward

:hidden:`Loss`
~~~~~~~~~~~~~~

:hidden:`RNNTLoss`
------------------

.. autoclass:: RNNTLoss

  .. automethod:: forward

:hidden:`Multi-channel`
~~~~~~~~~~~~~~~~~~~~~~~

:hidden:`PSD`
-------------

.. autoclass:: PSD

  .. automethod:: forward

:hidden:`MVDR`
--------------

.. autoclass:: MVDR

  .. automethod:: forward

:hidden:`RTFMVDR`
-----------------

.. autoclass:: RTFMVDR

  .. automethod:: forward

:hidden:`SoudenMVDR`
--------------------

.. autoclass:: SoudenMVDR

  .. automethod:: forward

References
~~~~~~~~~~

.. footbibliography::
