import math

import torch
import torchaudio.prototype.functional as F
from parameterized import parameterized
from torch.autograd import gradcheck
from torchaudio_unittest.common_utils import TestBaseMixin


class AutogradTestImpl(TestBaseMixin):
    @parameterized.expand(
        [
            (8000, (2, 3, 5, 7)),
            (8000, (8000, 1)),
        ]
    )
    def test_oscillator_bank(self, sample_rate, shape):
        numel = math.prod(shape)

        # use 1.9 instead of 2 so as to include values above nyquist frequency
        fmax = sample_rate / 1.9
        freq = torch.linspace(-fmax, fmax, numel, dtype=self.dtype, device=self.device, requires_grad=True).reshape(
            shape
        )
        amps = torch.linspace(-5, 5, numel, dtype=self.dtype, device=self.device, requires_grad=True).reshape(shape)

        assert gradcheck(F.oscillator_bank, (freq, amps, sample_rate))

    def test_extend_pitch(self):
        num_frames, num_pitches = 5, 7
        input = torch.ones((num_frames, 1), device=self.device, dtype=self.dtype, requires_grad=True)
        pattern = torch.linspace(1, num_pitches, num_pitches, device=self.device, dtype=self.dtype, requires_grad=True)

        assert gradcheck(F.extend_pitch, (input, num_pitches))
        assert gradcheck(F.extend_pitch, (input, pattern))

    def test_sinc_ir(self):
        cutoff = torch.tensor([0, 0.5, 1.0], device=self.device, dtype=self.dtype, requires_grad=True)
        assert gradcheck(F.sinc_impulse_response, (cutoff, 513, False))
        assert gradcheck(F.sinc_impulse_response, (cutoff, 513, True))

    def test_freq_ir(self):
        mags = torch.tensor([0, 0.5, 1.0], device=self.device, dtype=self.dtype, requires_grad=True)
        assert gradcheck(F.frequency_impulse_response, (mags,))

    def test_filter_waveform(self):
        waveform = torch.rand(3, 1, 2, 10, device=self.device, dtype=self.dtype, requires_grad=True)
        filters = torch.rand(3, 2, device=self.device, dtype=self.dtype, requires_grad=True)
        assert gradcheck(F.filter_waveform, (waveform, filters))

    def test_exp_sigmoid_input(self):
        input = torch.linspace(-5, 5, 20, device=self.device, dtype=self.dtype, requires_grad=True)
        exponent = 10.0
        max_value = 2.0
        threshold = 1e-7
        assert gradcheck(F.exp_sigmoid, (input, exponent, max_value, threshold))
