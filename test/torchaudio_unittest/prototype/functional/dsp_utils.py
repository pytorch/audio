import numpy as np


def oscillator_bank(
    frequencies,
    amplitudes,
    sample_rate: float,
    time_axis: int = -2,
):
    """Reference implementation of oscillator_bank"""
    invalid = np.abs(frequencies) >= sample_rate / 2
    if np.any(invalid):
        amplitudes = np.where(invalid, 0.0, amplitudes)
    pi2 = 2.0 * np.pi
    freqs = frequencies * pi2 / sample_rate % pi2
    phases = np.cumsum(freqs, axis=time_axis, dtype=freqs.dtype)

    waveform = amplitudes * np.sin(phases)
    return waveform


def sinc_ir(cutoff, window_size: int = 513, high_pass: bool = False):
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


def exp_sigmoid(
    input: np.ndarray, exponent: float = 10.0, max_value: float = 2.0, threshold: float = 1e-7
) -> np.ndarray:
    """Exponential Sigmoid pointwise nonlinearity (Numpy version).
    Implements the equation:
    ``max_value`` * sigmoid(``input``) ** (log(``exponent``)) + ``threshold``

    The output has a range of [``threshold``, ``max_value``].
    ``exponent`` controls the slope of the output.

    Args:
        input (np.ndarray): Input array
        exponent (float, optional): Exponent. Controls the slope of the output
        max_value (float, optional): Maximum value of the output
        threshold (float, optional): Minimum value of the output

    Returns:
        np.ndarray: Exponential Sigmoid output. Shape: same as input

    """
    return max_value * (1 / (1 + np.exp(-input, dtype=input.dtype))) ** np.log(exponent, dtype=input.dtype) + threshold
