from functools import partial
from typing import Callable, Tuple

import torch
import torchaudio.functional as F
from parameterized import parameterized
from torch import Tensor
from torch.autograd import gradcheck, gradgradcheck
from torchaudio_unittest.common_utils import (
    get_spectrogram,
    get_whitenoise,
    nested_params,
    rnnt_utils,
    TestBaseMixin,
    use_deterministic_algorithms,
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
        x = get_whitenoise(sample_rate=22050, duration=0.01, n_channels=2)
        a = torch.tensor([0.7, 0.2, 0.6])
        b = torch.tensor([0.4, 0.2, 0.9])
        x.requires_grad = True
        self.assert_grad(F.lfilter, (x, a, b), enable_all_grad=False)

    def test_lfilter_a(self):
        x = get_whitenoise(sample_rate=22050, duration=0.01, n_channels=2)
        a = torch.tensor([0.7, 0.2, 0.6])
        b = torch.tensor([0.4, 0.2, 0.9])
        a.requires_grad = True
        self.assert_grad(F.lfilter, (x, a, b), enable_all_grad=False)

    def test_lfilter_b(self):
        x = get_whitenoise(sample_rate=22050, duration=0.01, n_channels=2)
        a = torch.tensor([0.7, 0.2, 0.6])
        b = torch.tensor([0.4, 0.2, 0.9])
        b.requires_grad = True
        self.assert_grad(F.lfilter, (x, a, b), enable_all_grad=False)

    def test_lfilter_all_inputs(self):
        x = get_whitenoise(sample_rate=22050, duration=0.01, n_channels=2)
        a = torch.tensor([0.7, 0.2, 0.6])
        b = torch.tensor([0.4, 0.2, 0.9])
        self.assert_grad(F.lfilter, (x, a, b))

    def test_lfilter_filterbanks(self):
        x = get_whitenoise(sample_rate=22050, duration=0.01, n_channels=3)
        a = torch.tensor([[0.7, 0.2, 0.6], [0.8, 0.2, 0.9]])
        b = torch.tensor([[0.4, 0.2, 0.9], [0.7, 0.2, 0.6]])
        self.assert_grad(partial(F.lfilter, batching=False), (x, a, b))

    def test_lfilter_batching(self):
        x = get_whitenoise(sample_rate=22050, duration=0.01, n_channels=2)
        a = torch.tensor([[0.7, 0.2, 0.6], [0.8, 0.2, 0.9]])
        b = torch.tensor([[0.4, 0.2, 0.9], [0.7, 0.2, 0.6]])
        self.assert_grad(F.lfilter, (x, a, b))

    def test_filtfilt_a(self):
        x = get_whitenoise(sample_rate=22050, duration=0.01, n_channels=2)
        a = torch.tensor([0.7, 0.2, 0.6])
        b = torch.tensor([0.4, 0.2, 0.9])
        a.requires_grad = True
        with use_deterministic_algorithms(True, False):
            self.assert_grad(F.filtfilt, (x, a, b), enable_all_grad=False)

    def test_filtfilt_b(self):
        x = get_whitenoise(sample_rate=22050, duration=0.01, n_channels=2)
        a = torch.tensor([0.7, 0.2, 0.6])
        b = torch.tensor([0.4, 0.2, 0.9])
        b.requires_grad = True
        with use_deterministic_algorithms(True, False):
            self.assert_grad(F.filtfilt, (x, a, b), enable_all_grad=False)

    def test_filtfilt_all_inputs(self):
        x = get_whitenoise(sample_rate=22050, duration=0.01, n_channels=2)
        a = torch.tensor([0.7, 0.2, 0.6])
        b = torch.tensor([0.4, 0.2, 0.9])
        with use_deterministic_algorithms(True, False):
            self.assert_grad(F.filtfilt, (x, a, b))

    def test_filtfilt_batching(self):
        x = get_whitenoise(sample_rate=22050, duration=0.01, n_channels=2)
        a = torch.tensor([[0.7, 0.2, 0.6], [0.8, 0.2, 0.9]])
        b = torch.tensor([[0.4, 0.2, 0.9], [0.7, 0.2, 0.6]])
        with use_deterministic_algorithms(True, False):
            self.assert_grad(F.filtfilt, (x, a, b))

    def test_biquad(self):
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
        sr = 22050
        x = get_whitenoise(sample_rate=sr, duration=0.01, n_channels=1)
        central_freq = torch.tensor(central_freq)
        Q = torch.tensor(Q)
        self.assert_grad(F.bandreject_biquad, (x, sr, central_freq, Q))

    def test_deemph_biquad(self):
        x = get_whitenoise(sample_rate=22050, duration=0.01, n_channels=1)
        self.assert_grad(F.deemph_biquad, (x, 44100))

    def test_flanger(self):
        x = get_whitenoise(sample_rate=8000, duration=0.01, n_channels=1)
        self.assert_grad(F.flanger, (x, 44100))

    def test_gain(self):
        x = get_whitenoise(sample_rate=8000, duration=0.01, n_channels=1)
        self.assert_grad(F.gain, (x, 1.1))

    def test_overdrive(self):
        x = get_whitenoise(sample_rate=8000, duration=0.01, n_channels=1)
        self.assert_grad(F.gain, (x,))

    @parameterized.expand([(True,), (False,)])
    def test_phaser(self, sinusoidal):
        sr = 8000
        x = get_whitenoise(sample_rate=sr, duration=0.01, n_channels=1)
        self.assert_grad(F.phaser, (x, sr, sinusoidal))

    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    def test_psd(self, use_mask):
        specgram = torch.rand(4, 10, 5, dtype=torch.cfloat)
        if use_mask:
            mask = torch.rand(10, 5)
        else:
            mask = None
        self.assert_grad(F.psd, (specgram, mask))

    def test_mvdr_weights_souden(self):
        channel = 4
        n_fft_bin = 5
        psd_speech = torch.rand(n_fft_bin, channel, channel, dtype=torch.cfloat)
        psd_noise = torch.rand(n_fft_bin, channel, channel, dtype=torch.cfloat)
        self.assert_grad(F.mvdr_weights_souden, (psd_speech, psd_noise, 0))

    def test_mvdr_weights_souden_with_tensor(self):
        channel = 4
        n_fft_bin = 5
        psd_speech = torch.rand(n_fft_bin, channel, channel, dtype=torch.cfloat)
        psd_noise = torch.rand(n_fft_bin, channel, channel, dtype=torch.cfloat)
        reference_channel = torch.zeros(channel)
        reference_channel[0].fill_(1)
        self.assert_grad(F.mvdr_weights_souden, (psd_speech, psd_noise, reference_channel))

    def test_mvdr_weights_rtf(self):
        batch_size = 2
        channel = 4
        n_fft_bin = 10
        rtf = torch.rand(batch_size, n_fft_bin, channel, dtype=self.complex_dtype)
        psd_noise = torch.rand(batch_size, n_fft_bin, channel, channel, dtype=self.complex_dtype)
        self.assert_grad(F.mvdr_weights_rtf, (rtf, psd_noise, 0))

    def test_mvdr_weights_rtf_with_tensor(self):
        batch_size = 2
        channel = 4
        n_fft_bin = 10
        rtf = torch.rand(batch_size, n_fft_bin, channel, dtype=self.complex_dtype)
        psd_noise = torch.rand(batch_size, n_fft_bin, channel, channel, dtype=self.complex_dtype)
        reference_channel = torch.zeros(batch_size, channel)
        reference_channel[..., 0].fill_(1)
        self.assert_grad(F.mvdr_weights_rtf, (rtf, psd_noise, reference_channel))

    @parameterized.expand(
        [
            (1, True),
            (3, False),
        ]
    )
    def test_rtf_power(self, n_iter, diagonal_loading):
        channel = 4
        n_fft_bin = 5
        psd_speech = torch.rand(n_fft_bin, channel, channel, dtype=torch.cfloat)
        psd_noise = torch.rand(n_fft_bin, channel, channel, dtype=torch.cfloat)
        self.assert_grad(F.rtf_power, (psd_speech, psd_noise, 0, n_iter, diagonal_loading))

    @parameterized.expand(
        [
            (1, True),
            (3, False),
        ]
    )
    def test_rtf_power_with_tensor(self, n_iter, diagonal_loading):
        channel = 4
        n_fft_bin = 5
        psd_speech = torch.rand(n_fft_bin, channel, channel, dtype=torch.cfloat)
        psd_noise = torch.rand(n_fft_bin, channel, channel, dtype=torch.cfloat)
        reference_channel = torch.zeros(channel)
        reference_channel[0].fill_(1)
        self.assert_grad(F.rtf_power, (psd_speech, psd_noise, reference_channel, n_iter, diagonal_loading))

    def test_apply_beamforming(self):
        sr = 8000
        n_fft = 400
        batch_size, num_channels = 2, 3
        n_fft_bin = n_fft // 2 + 1
        x = get_whitenoise(sample_rate=sr, duration=0.05, n_channels=batch_size * num_channels)
        specgram = get_spectrogram(x, n_fft=n_fft, hop_length=100)
        specgram = specgram.view(batch_size, num_channels, n_fft_bin, specgram.size(-1))
        beamform_weights = torch.rand(batch_size, n_fft_bin, num_channels, dtype=torch.cfloat)
        self.assert_grad(F.apply_beamforming, (beamform_weights, specgram))

    @nested_params(
        ["convolve", "fftconvolve"],
        ["full", "valid", "same"],
    )
    def test_convolve(self, fn, mode):
        leading_dims = (4, 3, 2)
        L_x, L_y = 23, 40
        x = torch.rand(*leading_dims, L_x, dtype=self.dtype, device=self.device)
        y = torch.rand(*leading_dims, L_y, dtype=self.dtype, device=self.device)
        self.assert_grad(getattr(F, fn), (x, y, mode))

    def test_add_noise(self):
        leading_dims = (5, 2, 3)
        L = 51
        waveform = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device)
        noise = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device)
        lengths = torch.rand(*leading_dims, dtype=self.dtype, device=self.device)
        snr = torch.rand(*leading_dims, dtype=self.dtype, device=self.device) * 10
        self.assert_grad(F.add_noise, (waveform, noise, snr, lengths))

    def test_speed(self):
        leading_dims = (3, 2)
        T = 200
        waveform = torch.rand(*leading_dims, T, dtype=self.dtype, device=self.device, requires_grad=True)
        lengths = torch.randint(1, T, leading_dims, dtype=self.dtype, device=self.device)
        self.assert_grad(F.speed, (waveform, 1000, 1.1, lengths), enable_all_grad=False)

    def test_preemphasis(self):
        waveform = torch.rand(3, 2, 100, device=self.device, dtype=self.dtype, requires_grad=True)
        coeff = 0.9
        self.assert_grad(F.preemphasis, (waveform, coeff))

    def test_deemphasis(self):
        waveform = torch.rand(3, 2, 100, device=self.device, dtype=self.dtype, requires_grad=True)
        coeff = 0.9
        self.assert_grad(F.deemphasis, (waveform, coeff))


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
