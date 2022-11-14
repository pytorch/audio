from ._dsp import adsr_envelope, extend_pitch, oscillator_bank
from .functional import add_noise, barkscale_fbanks, convolve, fftconvolve

__all__ = [
    "add_noise",
    "adsr_envelope",
    "barkscale_fbanks",
    "convolve",
    "extend_pitch",
    "fftconvolve",
    "oscillator_bank",
]
