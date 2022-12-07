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


def sinc_ir(cutoff: ArrayLike, window_size: int = 513, high_pass: bool = False):
    if window_size % 2 == 0:
        raise ValueError(f"`window_size` must be odd. Given: {window_size}")
    half = window_size // 2
    dtype = cutoff.dtype
    idx = np.linspace(-half, half, window_size, dtype=dtype)

    filt = np.sinc(cutoff[..., None] * idx[None, ...])
    filt *= np.hamming(window_size).astype(dtype)[None, ...]
    filt /= np.abs(filt.sum(axis=-1, keepdims=True))

    if high_pass:
        filt *= -1
        filt[..., half] = 1.0 + filt[..., half]
    return filt


def freq_ir(magnitudes):
    ir = np.fft.fftshift(np.fft.irfft(magnitudes), axes=-1)
    window = np.hanning(ir.shape[-1])
    return (ir * window).astype(magnitudes.dtype)
