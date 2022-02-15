from functools import partial
from typing import Callable, Tuple

import torch
import torchaudio.functional as F
from parameterized import parameterized
from torch import Tensor
from torch.autograd import gradcheck, gradgradcheck
from torchaudio_unittest.common_utils import (
    TestBaseMixin,
    get_whitenoise,
    rnnt_utils,
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
                i = i.to(dtype=self.complex_dtype if i.is_complex() else self.dtype, device=self.device)
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

    def test_lfilter_filterbanks(self):
        torch.random.manual_seed(2434)
        x = get_whitenoise(sample_rate=22050, duration=0.01, n_channels=3)
        a = torch.tensor([[0.7, 0.2, 0.6], [0.8, 0.2, 0.9]])
        b = torch.tensor([[0.4, 0.2, 0.9], [0.7, 0.2, 0.6]])
        self.assert_grad(partial(F.lfilter, batching=False), (x, a, b))

    def test_lfilter_batching(self):
        torch.random.manual_seed(2434)
        x = get_whitenoise(sample_rate=22050, duration=0.01, n_channels=2)
        a = torch.tensor([[0.7, 0.2, 0.6], [0.8, 0.2, 0.9]])
        b = torch.tensor([[0.4, 0.2, 0.9], [0.7, 0.2, 0.6]])
        self.assert_grad(F.lfilter, (x, a, b))

    def test_filtfilt_a(self):
        torch.random.manual_seed(2434)
        x = get_whitenoise(sample_rate=22050, duration=0.01, n_channels=2)
        a = torch.tensor([0.7, 0.2, 0.6])
        b = torch.tensor([0.4, 0.2, 0.9])
        a.requires_grad = True
        self.assert_grad(F.filtfilt, (x, a, b), enable_all_grad=False)

    def test_filtfilt_b(self):
        torch.random.manual_seed(2434)
        x = get_whitenoise(sample_rate=22050, duration=0.01, n_channels=2)
        a = torch.tensor([0.7, 0.2, 0.6])
        b = torch.tensor([0.4, 0.2, 0.9])
        b.requires_grad = True
        self.assert_grad(F.filtfilt, (x, a, b), enable_all_grad=False)

    def test_filtfilt_all_inputs(self):
        torch.random.manual_seed(2434)
        x = get_whitenoise(sample_rate=22050, duration=0.01, n_channels=2)
        a = torch.tensor([0.7, 0.2, 0.6])
        b = torch.tensor([0.4, 0.2, 0.9])
        self.assert_grad(F.filtfilt, (x, a, b))

    def test_filtfilt_batching(self):
        torch.random.manual_seed(2434)
        x = get_whitenoise(sample_rate=22050, duration=0.01, n_channels=2)
        a = torch.tensor([[0.7, 0.2, 0.6], [0.8, 0.2, 0.9]])
        b = torch.tensor([[0.4, 0.2, 0.9], [0.7, 0.2, 0.6]])
        self.assert_grad(F.filtfilt, (x, a, b))

    def test_biquad(self):
        torch.random.manual_seed(2434)
        x = get_whitenoise(sample_rate=22050, duration=0.01, n_channels=1)
        a = torch.tensor([0.7, 0.2, 0.6])
        b = torch.tensor([0.4, 0.2, 0.9])
        self.assert_grad(F.biquad, (x, b[0], b[1], b[2], a[0], a[1], a[2]))

    @parameterized.expand(
        [
            (800, 0.7, True),
            (800, 0.7, False),
        ]
    )
    def test_band_biquad(self, central_freq, Q, noise):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.01, n_channels=1)
        central_freq = torch.tensor(central_freq)
        Q = torch.tensor(Q)
        self.assert_grad(F.band_biquad, (x, sr, central_freq, Q, noise))

    @parameterized.expand(
        [
            (800, 0.7, 10),
            (800, 0.7, -10),
        ]
    )
    def test_bass_biquad(self, central_freq, Q, gain):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.01, n_channels=1)
        central_freq = torch.tensor(central_freq)
        Q = torch.tensor(Q)
        gain = torch.tensor(gain)
        self.assert_grad(F.bass_biquad, (x, sr, gain, central_freq, Q))

    @parameterized.expand(
        [
            (3000, 0.7, 10),
            (3000, 0.7, -10),
        ]
    )
    def test_treble_biquad(self, central_freq, Q, gain):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.01, n_channels=1)
        central_freq = torch.tensor(central_freq)
        Q = torch.tensor(Q)
        gain = torch.tensor(gain)
        self.assert_grad(F.treble_biquad, (x, sr, gain, central_freq, Q))

    @parameterized.expand(
        [
            (
                800,
                0.7,
            ),
        ]
    )
    def test_allpass_biquad(self, central_freq, Q):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.01, n_channels=1)
        central_freq = torch.tensor(central_freq)
        Q = torch.tensor(Q)
        self.assert_grad(F.allpass_biquad, (x, sr, central_freq, Q))

    @parameterized.expand(
        [
            (
                800,
                0.7,
            ),
        ]
    )
    def test_lowpass_biquad(self, cutoff_freq, Q):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.01, n_channels=1)
        cutoff_freq = torch.tensor(cutoff_freq)
        Q = torch.tensor(Q)
        self.assert_grad(F.lowpass_biquad, (x, sr, cutoff_freq, Q))

    @parameterized.expand(
        [
            (
                800,
                0.7,
            ),
        ]
    )
    def test_highpass_biquad(self, cutoff_freq, Q):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.01, n_channels=1)
        cutoff_freq = torch.tensor(cutoff_freq)
        Q = torch.tensor(Q)
        self.assert_grad(F.highpass_biquad, (x, sr, cutoff_freq, Q))

    @parameterized.expand(
        [
            (800, 0.7, True),
            (800, 0.7, False),
        ]
    )
    def test_bandpass_biquad(self, central_freq, Q, const_skirt_gain):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.01, n_channels=1)
        central_freq = torch.tensor(central_freq)
        Q = torch.tensor(Q)
        self.assert_grad(F.bandpass_biquad, (x, sr, central_freq, Q, const_skirt_gain))

    @parameterized.expand(
        [
            (800, 0.7, 10),
            (800, 0.7, -10),
        ]
    )
    def test_equalizer_biquad(self, central_freq, Q, gain):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.01, n_channels=1)
        central_freq = torch.tensor(central_freq)
        Q = torch.tensor(Q)
        gain = torch.tensor(gain)
        self.assert_grad(F.equalizer_biquad, (x, sr, central_freq, gain, Q))

    @parameterized.expand(
        [
            (
                800,
                0.7,
            ),
        ]
    )
    def test_bandreject_biquad(self, central_freq, Q):
        torch.random.manual_seed(2434)
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.01, n_channels=1)
        central_freq = torch.tensor(central_freq)
        Q = torch.tensor(Q)
        self.assert_grad(F.bandreject_biquad, (x, sr, central_freq, Q))

    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    def test_compute_power_spectral_density_matrix(self, use_mask):
        torch.random.manual_seed(2434)
        specgram = torch.rand(4, 10, 5, dtype=torch.cfloat)
        if use_mask:
            mask = torch.rand(10, 5)
        else:
            mask = None
        self.assert_grad(F.compute_power_spectral_density_matrix, (specgram, mask))


class AutogradFloat32(TestBaseMixin):
    def assert_grad(
        self,
        transform: Callable[..., Tensor],
        inputs: Tuple[torch.Tensor],
        enable_all_grad: bool = True,
    ):
        inputs_ = []
        for i in inputs:
            if torch.is_tensor(i):
                i = i.to(dtype=self.dtype, device=self.device)
                if enable_all_grad:
                    i.requires_grad = True
            inputs_.append(i)
        # gradcheck with float32 requires higher atol and epsilon
        assert gradcheck(transform, inputs, eps=1e-3, atol=1e-3, nondet_tol=0.0)

    @parameterized.expand(
        [
            (rnnt_utils.get_B1_T10_U3_D4_data,),
            (rnnt_utils.get_B2_T4_U3_D3_data,),
            (rnnt_utils.get_B1_T2_U3_D5_data,),
        ]
    )
    def test_rnnt_loss(self, data_func):
        def get_data(data_func, device):
            data = data_func()
            if type(data) == tuple:
                data = data[0]
            return data

        data = get_data(data_func, self.device)
        inputs = (
            data["logits"].to(torch.float32),  # logits
            data["targets"],  # targets
            data["logit_lengths"],  # logit_lengths
            data["target_lengths"],  # target_lengths
            data["blank"],  # blank
            -1,  # clamp
        )

        self.assert_grad(F.rnnt_loss, inputs, enable_all_grad=False)
