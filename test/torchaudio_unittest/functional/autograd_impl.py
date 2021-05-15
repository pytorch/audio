from typing import Callable, Tuple
import torch
from parameterized import parameterized
from torch import Tensor
import torchaudio.functional as F
from torch.autograd import gradcheck, gradgradcheck
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
                i = i.to(dtype=self.dtype, device=self.device)
                if enable_all_grad:
                    i.requires_grad = True
            inputs_.append(i)
        assert gradcheck(transform, inputs_)
        assert gradgradcheck(transform, inputs_)

    def test_lfilter_x(self):
        torch.random.manual_seed(2434)
        x = get_whitenoise(sample_rate=22050, duration=0.01, n_channels=2)
        a = torch.tensor([0.7, 0.2, 0.6])
        b = torch.tensor([0.4, 0.2, 0.9])
        x.requires_grad = True
        self.assert_grad(F.lfilter, (x, a, b), enable_all_grad=False)

    def test_lfilter_a(self):
        torch.random.manual_seed(2434)
        x = get_whitenoise(sample_rate=22050, duration=0.01, n_channels=2)
        a = torch.tensor([0.7, 0.2, 0.6])
        b = torch.tensor([0.4, 0.2, 0.9])
        a.requires_grad = True
        self.assert_grad(F.lfilter, (x, a, b), enable_all_grad=False)

    def test_lfilter_b(self):
        torch.random.manual_seed(2434)
        x = get_whitenoise(sample_rate=22050, duration=0.01, n_channels=2)
        a = torch.tensor([0.7, 0.2, 0.6])
        b = torch.tensor([0.4, 0.2, 0.9])
        b.requires_grad = True
        self.assert_grad(F.lfilter, (x, a, b), enable_all_grad=False)

    def test_lfilter_all_inputs(self):
        torch.random.manual_seed(2434)
        x = get_whitenoise(sample_rate=22050, duration=0.01, n_channels=2)
        a = torch.tensor([0.7, 0.2, 0.6])
        b = torch.tensor([0.4, 0.2, 0.9])
        self.assert_grad(F.lfilter, (x, a, b))

    def test_biquad(self):
        torch.random.manual_seed(2434)
        x = get_whitenoise(sample_rate=22050, duration=0.01, n_channels=1)
        a = torch.tensor([0.7, 0.2, 0.6])
        b = torch.tensor([0.4, 0.2, 0.9])
        self.assert_grad(F.biquad, (x, b[0], b[1], b[2], a[0], a[1], a[2]))

    @parameterized.expand([
        (800, 0.7, True),
        (800, 0.7, False),
    ])
    def test_band_biquad(self, central_freq, Q, noise):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.01, n_channels=1)
        central_freq = torch.tensor(central_freq)
        Q = torch.tensor(Q)
        self.assert_grad(F.band_biquad, (x, sr, central_freq, Q, noise))

    @parameterized.expand([
        (800, 0.7, 10),
        (800, 0.7, -10),
    ])
    def test_bass_biquad(self, central_freq, Q, gain):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.01, n_channels=1)
        central_freq = torch.tensor(central_freq)
        Q = torch.tensor(Q)
        gain = torch.tensor(gain)
        self.assert_grad(F.bass_biquad, (x, sr, gain, central_freq, Q))

    @parameterized.expand([
        (3000, 0.7, 10),
        (3000, 0.7, -10),

    ])
    def test_treble_biquad(self, central_freq, Q, gain):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.01, n_channels=1)
        central_freq = torch.tensor(central_freq)
        Q = torch.tensor(Q)
        gain = torch.tensor(gain)
        self.assert_grad(F.treble_biquad, (x, sr, gain, central_freq, Q))

    @parameterized.expand([
        (800, 0.7, ),
    ])
    def test_allpass_biquad(self, central_freq, Q):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.01, n_channels=1)
        central_freq = torch.tensor(central_freq)
        Q = torch.tensor(Q)
        self.assert_grad(F.allpass_biquad, (x, sr, central_freq, Q))

    @parameterized.expand([
        (800, 0.7, ),
    ])
    def test_lowpass_biquad(self, cutoff_freq, Q):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.01, n_channels=1)
        cutoff_freq = torch.tensor(cutoff_freq)
        Q = torch.tensor(Q)
        self.assert_grad(F.lowpass_biquad, (x, sr, cutoff_freq, Q))

    @parameterized.expand([
        (800, 0.7, ),
    ])
    def test_highpass_biquad(self, cutoff_freq, Q):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.01, n_channels=1)
        cutoff_freq = torch.tensor(cutoff_freq)
        Q = torch.tensor(Q)
        self.assert_grad(F.highpass_biquad, (x, sr, cutoff_freq, Q))

    @parameterized.expand([
        (800, 0.7, True),
        (800, 0.7, False),
    ])
    def test_bandpass_biquad(self, central_freq, Q, const_skirt_gain):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.01, n_channels=1)
        central_freq = torch.tensor(central_freq)
        Q = torch.tensor(Q)
        self.assert_grad(F.bandpass_biquad, (x, sr, central_freq, Q, const_skirt_gain))

    @parameterized.expand([
        (800, 0.7, 10),
        (800, 0.7, -10),
    ])
    def test_equalizer_biquad(self, central_freq, Q, gain):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.01, n_channels=1)
        central_freq = torch.tensor(central_freq)
        Q = torch.tensor(Q)
        gain = torch.tensor(gain)
        self.assert_grad(F.equalizer_biquad, (x, sr, central_freq, gain, Q))

    @parameterized.expand([
        (800, 0.7, ),
    ])
    def test_bandreject_biquad(self, central_freq, Q):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.01, n_channels=1)
        central_freq = torch.tensor(central_freq)
        Q = torch.tensor(Q)
        self.assert_grad(F.bandreject_biquad, (x, sr, central_freq, Q))
