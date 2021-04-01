from typing import Callable, List, Tuple
import torch
from torch import functional
from torch.tensor import Tensor
import torchaudio.functional as F
from torch.autograd import gradcheck
from torchaudio_unittest.common_utils import (
    TestBaseMixin,
    get_whitenoise,
)


class Autograd(TestBaseMixin):
    def assert_grad(
            self,
            transform: Callable[..., Tensor],
            inputs: Tuple[torch.Tensor],
            *,
            enable_all_grad: bool = True,
    ):
        inputs_ = []
        for i in inputs:
            if torch.is_tensor(i):
                if enable_all_grad:
                    i.requires_grad = True
                i = i.to(dtype=self.dtype, device=self.device)
            inputs_.append(i)
        assert gradcheck(transform, inputs_)

    def test_lfilter_x(self):
        torch.random.manual_seed(2434)
        x = get_whitenoise(sample_rate=22050, duration=0.025, n_channels=2)
        a = torch.tensor([0.7, 0.2, 0.6])
        b = torch.tensor([0.4, 0.2, 0.9])
        x.requires_grad = True
        self.assert_grad(F.lfilter, (x, a, b), enable_all_grad=False)

    def test_lfilter_a(self):
        torch.random.manual_seed(2434)
        x = get_whitenoise(sample_rate=22050, duration=0.05, n_channels=2)
        a = torch.tensor([0.7, 0.2, 0.6])
        b = torch.tensor([0.4, 0.2, 0.9])
        a.requires_grad = True
        self.assert_grad(F.lfilter, (x, a, b), enable_all_grad=False)

    def test_lfilter_b(self):
        torch.random.manual_seed(2434)
        x = get_whitenoise(sample_rate=22050, duration=0.05, n_channels=2)
        a = torch.tensor([0.7, 0.2, 0.6])
        b = torch.tensor([0.4, 0.2, 0.9])
        b.requires_grad = True
        self.assert_grad(F.lfilter, (x, a, b), enable_all_grad=False)

    def test_lfilter_all_inputs(self):
        torch.random.manual_seed(2434)
        x = get_whitenoise(sample_rate=22050, duration=0.05, n_channels=2)
        a = torch.tensor([0.7, 0.2, 0.6])
        b = torch.tensor([0.4, 0.2, 0.9])
        self.assert_grad(F.lfilter, (x, a, b))

    def test_biquad(self):
        torch.random.manual_seed(2434)
        x = get_whitenoise(sample_rate=22050, duration=0.05, n_channels=2)
        a = torch.tensor([0.7, 0.2, 0.6])
        b = torch.tensor([0.4, 0.2, 0.9])
        self.assert_grad(F.biquad, (x, b[0], b[1], b[2], a[0], a[1], a[2]))

    def test_band_biquad(self):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.05, n_channels=2)
        central_freq = torch.tensor(800.)
        Q = torch.tensor(0.7)
        self.assert_grad(F.band_biquad, (x, sr, central_freq, Q))

    def test_band_biquad_with_noise(self):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.05, n_channels=2)
        central_freq = torch.tensor(800.)
        Q = torch.tensor(0.7)
        self.assert_grad(F.band_biquad, (x, sr, central_freq, Q, True))

    def test_bass_biquad(self):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.05, n_channels=2)
        central_freq = torch.tensor(100.)
        Q = torch.tensor(0.7)
        gain = torch.tensor(10.)
        self.assert_grad(F.bass_biquad, (x, sr, gain, central_freq, Q))

    def test_treble_biquad(self):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.05, n_channels=2)
        central_freq = torch.tensor(3000.)
        Q = torch.tensor(0.7)
        gain = torch.tensor(10.)
        self.assert_grad(F.treble_biquad, (x, sr, gain, central_freq, Q))

    def test_allpass_biquad(self):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.05, n_channels=2)
        central_freq = torch.tensor(800.)
        Q = torch.tensor(0.7)
        self.assert_grad(F.allpass_biquad, (x, sr, central_freq, Q))

    def test_lowpass_biquad(self):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.05, n_channels=2)
        cutoff_freq = torch.tensor(800.)
        Q = torch.tensor(0.7)
        self.assert_grad(F.lowpass_biquad, (x, sr, cutoff_freq, Q))

    def test_highpass_biquad(self):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.05, n_channels=2)
        cutoff_freq = torch.tensor(800.)
        Q = torch.tensor(0.7)
        self.assert_grad(F.highpass_biquad, (x, sr, cutoff_freq, Q))

    def test_bandpass_biquad(self):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.05, n_channels=2)
        central_freq = torch.tensor(800.)
        Q = torch.tensor(0.7)
        self.assert_grad(F.bandpass_biquad, (x, sr, central_freq, Q))

    def test_bandpass_biquad_with_const_skirt_gain(self):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.05, n_channels=2)
        central_freq = torch.tensor(800.)
        Q = torch.tensor(0.7)
        self.assert_grad(F.bandpass_biquad, (x, sr, central_freq, Q, True))

    def test_equalizer_biquad(self):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.05, n_channels=2)
        central_freq = torch.tensor(800.)
        Q = torch.tensor(0.7)
        gain = torch.tensor(10.)
        self.assert_grad(F.equalizer_biquad, (x, sr, central_freq, gain, Q))

    def test_bandreject_biquad(self):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.05, n_channels=2)
        central_freq = torch.tensor(800.)
        Q = torch.tensor(0.7)
        self.assert_grad(F.bandreject_biquad, (x, sr, central_freq, Q))
