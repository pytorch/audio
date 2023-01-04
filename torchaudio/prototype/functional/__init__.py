from ._dsp import (
    adsr_envelope,
    extend_pitch,
    filter_waveform,
    frequency_impulse_response,
    oscillator_bank,
    sinc_impulse_response,
)
from .functional import add_noise, barkscale_fbanks, convolve, deemphasis, fftconvolve, preemphasis, speed


__all__ = [
    "add_noise",
    "adsr_envelope",
    "barkscale_fbanks",
    "convolve",
    "deemphasis",
    "extend_pitch",
    "fftconvolve",
    "filter_waveform",
    "frequency_impulse_response",
    "oscillator_bank",
    "preemphasis",
    "sinc_impulse_response",
    "speed",
]
