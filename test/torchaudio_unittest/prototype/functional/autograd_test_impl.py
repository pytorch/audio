import torch
import torchaudio.prototype.functional as F
from parameterized import parameterized
from torch.autograd import gradcheck, gradgradcheck
from torchaudio_unittest.common_utils import nested_params, TestBaseMixin


class AutogradTestImpl(TestBaseMixin):
    @nested_params(
        [F.convolve, F.fftconvolve],
        ["full", "valid", "same"],
    )
    def test_convolve(self, fn, mode):
        leading_dims = (4, 3, 2)
        L_x, L_y = 23, 40
        x = torch.rand(*leading_dims, L_x, dtype=self.dtype, device=self.device, requires_grad=True)
        y = torch.rand(*leading_dims, L_y, dtype=self.dtype, device=self.device, requires_grad=True)
        self.assertTrue(gradcheck(fn, (x, y, mode)))
        self.assertTrue(gradgradcheck(fn, (x, y, mode)))

    def test_add_noise(self):
        leading_dims = (5, 2, 3)
        L = 51

        waveform = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device, requires_grad=True)
        noise = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device, requires_grad=True)
        lengths = torch.rand(*leading_dims, dtype=self.dtype, device=self.device, requires_grad=True)
        snr = torch.rand(*leading_dims, dtype=self.dtype, device=self.device, requires_grad=True) * 10

        self.assertTrue(gradcheck(F.add_noise, (waveform, noise, lengths, snr)))
        self.assertTrue(gradgradcheck(F.add_noise, (waveform, noise, lengths, snr)))

    @parameterized.expand(
        [
            (8000, (2, 3, 5, 7)),
            (8000, (8000, 1)),
        ]
    )
    def test_oscillator_bank(self, sample_rate, shape):
        # can be replaced with math.prod when we drop 3.7 support
        def prod(iterable):
            ret = 1
            for item in iterable:
                ret *= item
            return ret

        numel = prod(shape)

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
