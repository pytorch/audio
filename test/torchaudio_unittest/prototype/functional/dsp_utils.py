import numpy as np
from numpy.typing import ArrayLike


def oscillator_bank(
    frequencies: ArrayLike,
    amplitudes: ArrayLike,
    sample_rate: float,
    time_axis: int = -2,
) -> ArrayLike:
    """Reference implementation of oscillator_bank"""
    invalid = np.abs(frequencies) >= sample_rate / 2
    if np.any(invalid):
        amplitudes = np.where(invalid, 0.0, amplitudes)
    pi2 = 2.0 * np.pi
    freqs = frequencies * pi2 / sample_rate % pi2
    phases = np.cumsum(freqs, axis=time_axis, dtype=freqs.dtype)

    waveform = amplitudes * np.sin(phases)
    return waveform
