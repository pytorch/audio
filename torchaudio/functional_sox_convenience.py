from __future__ import absolute_import, division, print_function, unicode_literals
import math
import torch
import torchaudio
from _torch_filtering import diff_eq as diffeq_cpp

__all__ = ["lowpass_biquad", "highpass_biquad", "biquad"]


def biquad(input_waveform, b0, b1, b2, a0, a1, a2):
    output_waveform = torch.zeros_like(input_waveform)
    assert input_waveform.dtype == torch.float32

    diffeq_cpp(
        input_waveform,
        output_waveform,
        torch.tensor([a0, a1, a2]),
        torch.tensor([b0, b1, b2]),
    )
    return output_waveform


def _dB2Linear(x):

    return math.exp(x * math.log(10) / 20.0)


def highpass_biquad(input_waveform, sr, cutoff_freq, Q=0.707):

    GAIN = 1
    w0 = 2 * math.pi * cutoff_freq / sr
    A = math.exp(GAIN / 40.0 * math.log(10))
    alpha = math.sin(w0) / 2 / Q
    mult = _dB2Linear(max(GAIN, 0))

    b0 = (1 + math.cos(w0)) / 2
    b1 = -1 - math.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha
    return biquad(input_waveform, b0, b1, b2, a0, a1, a2)


def lowpass_biquad(input_waveform, sr, cutoff_freq, Q=0.707):

    GAIN = 1
    w0 = 2 * math.pi * cutoff_freq / sr
    A = math.exp(GAIN / 40.0 * math.log(10))
    alpha = math.sin(w0) / 2 / Q
    mult = _dB2Linear(max(GAIN, 0))

    b0 = (1 - math.cos(w0)) / 2
    b1 = 1 - math.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha
    return biquad(input_waveform, b0, b1, b2, a0, a1, a2)


if __name__ == "__main__":
    pass
