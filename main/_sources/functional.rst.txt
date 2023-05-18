.. py:module:: torchaudio.functional

torchaudio.functional
=====================

.. currentmodule:: torchaudio.functional

Functions to perform common audio operations.

Utility
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   amplitude_to_DB
   DB_to_amplitude
   melscale_fbanks
   linear_fbanks
   create_dct
   mask_along_axis
   mask_along_axis_iid
   mu_law_encoding
   mu_law_decoding
   apply_codec
   resample
   loudness
   convolve
   fftconvolve
   add_noise
   preemphasis
   deemphasis
   speed
   frechet_distance

Forced Alignment
----------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   forced_align
   merge_tokens
   TokenSpan


Filtering
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   allpass_biquad
   band_biquad
   bandpass_biquad
   bandreject_biquad
   bass_biquad
   biquad
   contrast
   dcshift
   deemph_biquad
   dither
   equalizer_biquad
   filtfilt
   flanger
   gain
   highpass_biquad
   lfilter
   lowpass_biquad
   overdrive
   phaser
   riaa_biquad
   treble_biquad

Feature Extractions
-------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   vad
   spectrogram
   inverse_spectrogram
   griffinlim
   phase_vocoder
   pitch_shift
   compute_deltas
   detect_pitch_frequency
   sliding_window_cmn
   spectral_centroid

Multi-channel
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   psd
   mvdr_weights_souden
   mvdr_weights_rtf
   rtf_evd
   rtf_power
   apply_beamforming

Loss
----

.. autosummary::
   :toctree: generated
   :nosignatures:

   rnnt_loss

Metric
------

.. autosummary::
   :toctree: generated
   :nosignatures:

   edit_distance
