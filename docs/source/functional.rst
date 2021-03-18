.. role:: hidden
    :class: hidden-section

torchaudio.functional
=====================

.. currentmodule:: torchaudio.functional

Functions to perform common audio operations.

:hidden:`Utility`
~~~~~~~~~~~~~~~~~

amplitude_to_DB
---------------

.. autofunction:: amplitude_to_DB

DB_to_amplitude
---------------

.. autofunction:: DB_to_amplitude

create_fb_matrix
----------------

.. autofunction:: create_fb_matrix

create_dct
----------

.. autofunction:: create_dct

mask_along_axis
---------------

.. autofunction:: mask_along_axis

mask_along_axis_iid
-------------------

.. autofunction:: mask_along_axis_iid

mu_law_encoding
---------------

.. autofunction:: mu_law_encoding

mu_law_decoding
---------------

.. autofunction:: mu_law_decoding

apply_codec
-----------

.. autofunction:: apply_codec
		  
:hidden:`Complex Utility`
~~~~~~~~~~~~~~~~~~~~~~~~~

Utilities for pseudo complex tensor. This is not for the native complex dtype, such as `cfloat64`, but for tensors with real-value type and have extra dimension at the end for real and imaginary parts.

angle
-----

.. autofunction:: angle

complex_norm
------------

.. autofunction:: complex_norm


magphase
--------

.. autofunction:: magphase

:hidden:`Filtering`
~~~~~~~~~~~~~~~~~~~


allpass_biquad
--------------

.. autofunction:: allpass_biquad

band_biquad
-----------

.. autofunction:: band_biquad

bandpass_biquad
---------------

.. autofunction:: bandpass_biquad

bandreject_biquad
-----------------

.. autofunction:: bandreject_biquad

bass_biquad
-----------

.. autofunction:: bass_biquad

biquad
------

.. autofunction:: biquad

contrast
--------

.. autofunction:: contrast

dcshift
-------

.. autofunction:: dcshift

deemph_biquad
-------------

.. autofunction:: deemph_biquad


dither
------

.. autofunction:: dither

equalizer_biquad
----------------

.. autofunction:: equalizer_biquad

flanger
-------

.. autofunction:: flanger

gain
----

.. autofunction:: gain

highpass_biquad
---------------

.. autofunction:: highpass_biquad

lfilter
-------

.. autofunction:: lfilter

lowpass_biquad
--------------

.. autofunction:: lowpass_biquad

overdrive
---------

.. autofunction:: overdrive

phaser
------

.. autofunction:: phaser

riaa_biquad
-----------

.. autofunction:: riaa_biquad

treble_biquad
-------------

.. autofunction:: treble_biquad


vad
---

:hidden:`Feature Extractions`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vad

:hidden:`spectrogram`
---------------------

.. autofunction:: spectrogram

:hidden:`griffinlim`
--------------------

.. autofunction:: griffinlim

:hidden:`phase_vocoder`
-----------------------

.. autofunction:: phase_vocoder

:hidden:`compute_deltas`
------------------------

.. autofunction:: compute_deltas

:hidden:`detect_pitch_frequency`
--------------------------------

.. autofunction:: detect_pitch_frequency

:hidden:`sliding_window_cmn`
----------------------------

.. autofunction:: sliding_window_cmn

:hidden:`compute_kaldi_pitch`
-----------------------------

.. autofunction:: compute_kaldi_pitch

:hidden:`spectral_centroid`
---------------------------

.. autofunction:: spectral_centroid
