from __future__ import absolute_import, division, print_function, unicode_literals
import math
import torch
import torchaudio

__all__ = [
    "lowpass_biquad",
    "highpass_biquad",
    "biquad",
]


def biquad(waveform, b0, b1, b2, a0, a1, a2):
    # type: (Tensor, float, float, float, float, float, float) -> Tensor
    r"""Performs a biquad filter of input tensor.  Initial conditions set to 0.
    https://en.wikipedia.org/wiki/Digital_biquad_filter

    Args:
        waveform (torch.Tensor): audio waveform of dimension of `(n_channel, n_frames)`
        b0 (float): numerator coefficient of current input, x[n]
        b1 (float): numerator coefficient of input one time step ago x[n-1]
        b2 (float): numerator coefficient of input two time steps ago x[n-2]
        a0 (float): denominator coefficient of current output y[n], typically 1
        a1 (float): denominator coefficient of current output y[n-1]
        a2 (float): denominator coefficient of current output y[n-2]

    Returns:
        output_waveform (torch.Tensor): Dimension of `(n_channel, n_frames)`
    """

    assert(waveform.dtype == torch.float32)

    output_waveform = torchaudio.functional.lfilter(
        waveform, torch.tensor([a0, a1, a2]), torch.tensor([b0, b1, b2])
    )
    return output_waveform


def _dB2Linear(x):
    return math.exp(x * math.log(10) / 20.0)


def highpass_biquad(waveform, sample_rate, cutoff_freq, Q=0.707):
    # type: (Tensor, int, float, Optional[float]) -> Tensor
    r"""Designs biquad highpass filter and performs filtering.  Similar to SoX implementation.

    Args:
        waveform (torch.Tensor): audio waveform of dimension of `(n_channel, n_frames)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        cutoff_freq (float): filter cutoff frequency
        Q (float): https://en.wikipedia.org/wiki/Q_factor

    Returns:
        output_waveform (torch.Tensor): Dimension of `(n_channel, n_frames)`
    """

    GAIN = 1  # TBD - add as a parameter
    w0 = 2 * math.pi * cutoff_freq / sample_rate
    A = math.exp(GAIN / 40.0 * math.log(10))
    alpha = math.sin(w0) / 2 / Q
    mult = _dB2Linear(max(GAIN, 0))

    b0 = (1 + math.cos(w0)) / 2
    b1 = -1 - math.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha
    return biquad(waveform, b0, b1, b2, a0, a1, a2)


def lowpass_biquad(waveform, sample_rate, cutoff_freq, Q=0.707):
    # type: (Tensor, int, float, Optional[float]) -> Tensor
    r"""Designs biquad lowpass filter and performs filtering.  Similar to SoX implementation.

    Args:
        waveform (torch.Tensor): audio waveform of dimension of `(n_channel, n_frames)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        cutoff_freq (float): filter cutoff frequency
        Q (float): https://en.wikipedia.org/wiki/Q_factor

    Returns:
        output_waveform (torch.Tensor): Dimension of `(n_channel, n_frames)`
    """

    GAIN = 1
    w0 = 2 * math.pi * cutoff_freq / sample_rate
    A = math.exp(GAIN / 40.0 * math.log(10))
    alpha = math.sin(w0) / 2 / Q
    mult = _dB2Linear(max(GAIN, 0))

    b0 = (1 - math.cos(w0)) / 2
    b1 = 1 - math.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha
    return biquad(waveform, b0, b1, b2, a0, a1, a2)
