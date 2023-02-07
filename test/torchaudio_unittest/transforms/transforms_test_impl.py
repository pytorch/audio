import math
import random
from unittest.mock import patch

import numpy as np
import torch
import torchaudio.transforms as T
from parameterized import param, parameterized
from scipy import signal
from torchaudio.functional import lfilter, preemphasis
from torchaudio.functional.functional import _get_sinc_resample_kernel
from torchaudio_unittest.common_utils import get_spectrogram, get_whitenoise, nested_params, TestBaseMixin
from torchaudio_unittest.common_utils.psd_utils import psd_numpy


def _get_ratio(mat):
    return (mat.sum() / mat.numel()).item()


class TransformsTestBase(TestBaseMixin):
    def test_InverseMelScale(self):
        """Gauge the quality of InverseMelScale transform.

        As InverseMelScale is currently implemented with
        random initialization + iterative optimization,
        it is not practically possible to assert the difference between
        the estimated spectrogram and the original spectrogram as a whole.
        Estimated spectrogram has very huge descrepency locally.
        Thus in this test we gauge what percentage of elements are bellow
        certain tolerance.
        At the moment, the quality of estimated spectrogram is not good.
        When implementation is changed in a way it makes the quality even worse,
        this test will fail.
        """
        n_fft = 400
        power = 1
        n_mels = 64
        sample_rate = 8000

        n_stft = n_fft // 2 + 1

        # Generate reference spectrogram and input mel-scaled spectrogram
        expected = get_spectrogram(
            get_whitenoise(sample_rate=sample_rate, duration=1, n_channels=2), n_fft=n_fft, power=power
        ).to(self.device, self.dtype)
        input = T.MelScale(n_mels=n_mels, sample_rate=sample_rate, n_stft=n_stft).to(self.device, self.dtype)(expected)

        # Run transform
        transform = T.InverseMelScale(n_stft, n_mels=n_mels, sample_rate=sample_rate).to(self.device, self.dtype)
        result = transform(input)

        # Compare
        epsilon = 1e-60
        relative_diff = torch.abs((result - expected) / (expected + epsilon))

        for tol in [1e-1, 1e-3, 1e-5, 1e-10]:
            print(f"Ratio of relative diff smaller than {tol:e} is " f"{_get_ratio(relative_diff < tol)}")
        assert _get_ratio(relative_diff < 1e-1) > 0.2
        assert _get_ratio(relative_diff < 1e-3) > 5e-3
        assert _get_ratio(relative_diff < 1e-5) > 1e-5

    @nested_params(
        ["sinc_interp_hann", "sinc_interp_kaiser"],
        [16000, 44100],
    )
    def test_resample_identity(self, resampling_method, sample_rate):
        """When sampling rate is not changed, the transform returns an identical Tensor"""
        waveform = get_whitenoise(sample_rate=sample_rate, duration=1)

        resampler = T.Resample(sample_rate, sample_rate, resampling_method)
        resampled = resampler(waveform)
        self.assertEqual(waveform, resampled)

    @nested_params(
        ["sinc_interp_hann", "sinc_interp_kaiser"],
        [None, torch.float64],
    )
    def test_resample_cache_dtype(self, resampling_method, dtype):
        """Providing dtype changes the kernel cache dtype"""
        transform = T.Resample(16000, 44100, resampling_method, dtype=dtype)

        assert transform.kernel.dtype == dtype if dtype is not None else torch.float32

    @parameterized.expand(
        [
            param(n_fft=300, center=True, onesided=True),
            param(n_fft=400, center=True, onesided=False),
            param(n_fft=400, center=True, onesided=False),
            param(n_fft=300, center=True, onesided=False),
            param(n_fft=400, hop_length=10),
            param(n_fft=800, win_length=400, hop_length=20),
            param(n_fft=800, win_length=400, hop_length=20, normalized=True),
            param(),
            param(n_fft=400, pad=32),
            #   These tests do not work - cause runtime error
            #   See https://github.com/pytorch/pytorch/issues/62323
            #        param(n_fft=400, center=False, onesided=True),
            #        param(n_fft=400, center=False, onesided=False),
        ]
    )
    def test_roundtrip_spectrogram(self, **args):
        """Test the spectrogram + inverse spectrogram results in approximate identity."""

        waveform = get_whitenoise(sample_rate=8000, duration=0.5, dtype=self.dtype)

        s = T.Spectrogram(**args, power=None)
        inv_s = T.InverseSpectrogram(**args)
        transformed = s.forward(waveform)
        restored = inv_s.forward(transformed, length=waveform.shape[-1])
        self.assertEqual(waveform, restored, atol=1e-6, rtol=1e-6)

    @parameterized.expand(
        [
            param(0.5, 1, True, False),
            param(0.5, 1, None, False),
            param(1, 4, True, True),
            param(1, 6, None, True),
        ]
    )
    def test_psd(self, duration, channel, mask, multi_mask):
        """Providing dtype changes the kernel cache dtype"""
        transform = T.PSD(multi_mask)
        waveform = get_whitenoise(sample_rate=8000, duration=duration, n_channels=channel)
        spectrogram = get_spectrogram(waveform, n_fft=400)  # (channel, freq, time)
        spectrogram = spectrogram.to(torch.cdouble)
        if mask is not None:
            if multi_mask:
                mask = torch.rand(spectrogram.shape[-3:])
            else:
                mask = torch.rand(spectrogram.shape[-2:])
            psd_np = psd_numpy(spectrogram.detach().numpy(), mask.detach().numpy(), multi_mask)
        else:
            psd_np = psd_numpy(spectrogram.detach().numpy(), mask, multi_mask)
        psd = transform(spectrogram, mask)
        self.assertEqual(psd, psd_np, atol=1e-5, rtol=1e-5)

    @parameterized.expand(
        [
            param(torch.complex64),
            param(torch.complex128),
        ]
    )
    def test_mvdr(self, dtype):
        """Make sure the output dtype is the same as the input dtype"""
        transform = T.MVDR()
        waveform = get_whitenoise(sample_rate=8000, duration=0.5, n_channels=3)
        specgram = get_spectrogram(waveform, n_fft=400)  # (channel, freq, time)
        specgram = specgram.to(dtype)
        mask_s = torch.rand(specgram.shape[-2:])
        mask_n = torch.rand(specgram.shape[-2:])
        specgram_enhanced = transform(specgram, mask_s, mask_n)
        assert specgram_enhanced.dtype == dtype

    def test_pitch_shift_resample_kernel(self):
        """The resampling kernel in PitchShift is identical to what helper function generates.
        There should be no numerical difference caused by dtype conversion.
        """
        sample_rate = 8000
        trans = T.PitchShift(sample_rate=sample_rate, n_steps=4)
        trans.to(self.dtype).to(self.device)
        # dry run to initialize the kernel
        trans(torch.randn(2, 8000, dtype=self.dtype, device=self.device))

        expected, _ = _get_sinc_resample_kernel(
            trans.orig_freq, sample_rate, trans.gcd, device=self.device, dtype=self.dtype
        )
        self.assertEqual(trans.kernel, expected)

    @nested_params(
        [(10, 4), (4, 3, 1, 2), (2,), ()],
        [(100, 43), (21, 45)],
        ["full", "valid", "same"],
    )
    def test_convolve(self, leading_dims, lengths, mode):
        """Check that Convolve returns values identical to those that SciPy produces."""
        L_x, L_y = lengths

        x = torch.rand(*(leading_dims + (L_x,)), dtype=self.dtype, device=self.device)
        y = torch.rand(*(leading_dims + (L_y,)), dtype=self.dtype, device=self.device)

        convolve = T.Convolve(mode=mode).to(self.device)
        actual = convolve(x, y)

        num_signals = torch.tensor(leading_dims).prod() if leading_dims else 1
        x_reshaped = x.reshape((num_signals, L_x))
        y_reshaped = y.reshape((num_signals, L_y))
        expected = [
            signal.convolve(x_reshaped[i].detach().cpu().numpy(), y_reshaped[i].detach().cpu().numpy(), mode=mode)
            for i in range(num_signals)
        ]
        expected = torch.tensor(np.array(expected))
        expected = expected.reshape(leading_dims + (-1,))

        self.assertEqual(expected, actual)

    @nested_params(
        [(10, 4), (4, 3, 1, 2), (2,), ()],
        [(100, 43), (21, 45)],
        ["full", "valid", "same"],
    )
    def test_fftconvolve(self, leading_dims, lengths, mode):
        """Check that FFTConvolve returns values identical to those that SciPy produces."""
        L_x, L_y = lengths

        x = torch.rand(*(leading_dims + (L_x,)), dtype=self.dtype, device=self.device)
        y = torch.rand(*(leading_dims + (L_y,)), dtype=self.dtype, device=self.device)

        convolve = T.FFTConvolve(mode=mode).to(self.device)
        actual = convolve(x, y)

        expected = signal.fftconvolve(x.detach().cpu().numpy(), y.detach().cpu().numpy(), axes=-1, mode=mode)
        expected = torch.tensor(expected)

        self.assertEqual(expected, actual)

    def test_speed_identity(self):
        """speed of 1.0 does not alter input waveform and length"""
        leading_dims = (5, 4, 2)
        time = 1000
        waveform = torch.rand(*leading_dims, time)
        lengths = torch.randint(1, 1000, leading_dims)
        speed = T.Speed(1000, 1.0)
        actual_waveform, actual_lengths = speed(waveform, lengths)
        self.assertEqual(waveform, actual_waveform)
        self.assertEqual(lengths, actual_lengths)

    @nested_params(
        [0.8, 1.1, 1.2],
    )
    def test_speed_accuracy(self, factor):
        """sinusoidal waveform is properly compressed by factor"""
        n_to_trim = 20

        sample_rate = 1000
        freq = 2
        times = torch.arange(0, 5, 1.0 / sample_rate)
        waveform = torch.cos(2 * math.pi * freq * times).unsqueeze(0).to(self.device, self.dtype)
        lengths = torch.tensor([waveform.size(1)])

        speed = T.Speed(sample_rate, factor).to(self.device, self.dtype)
        output, output_lengths = speed(waveform, lengths)
        self.assertEqual(output.size(1), output_lengths[0])

        new_times = torch.arange(0, 5 / factor, 1.0 / sample_rate)
        expected_waveform = torch.cos(2 * math.pi * freq * factor * new_times).unsqueeze(0).to(self.device, self.dtype)

        self.assertEqual(
            expected_waveform[..., n_to_trim:-n_to_trim], output[..., n_to_trim:-n_to_trim], atol=1e-1, rtol=1e-4
        )

    def test_speed_perturbation(self):
        """sinusoidal waveform is properly compressed by sampled factors"""
        n_to_trim = 20

        sample_rate = 1000
        freq = 2
        times = torch.arange(0, 5, 1.0 / sample_rate)
        waveform = torch.cos(2 * math.pi * freq * times).unsqueeze(0).to(self.device, self.dtype)
        lengths = torch.tensor([waveform.size(1)])

        factors = [0.8, 1.1, 1.0]
        indices = random.choices(range(len(factors)), k=5)

        speed_perturb = T.SpeedPerturbation(sample_rate, factors).to(self.device, self.dtype)

        with patch("torch.randint", side_effect=indices):
            for idx in indices:
                output, output_lengths = speed_perturb(waveform, lengths)
                self.assertEqual(output.size(1), output_lengths[0])
                factor = factors[idx]
                new_times = torch.arange(0, 5 / factor, 1.0 / sample_rate)
                expected_waveform = (
                    torch.cos(2 * math.pi * freq * factor * new_times).unsqueeze(0).to(self.device, self.dtype)
                )
                self.assertEqual(
                    expected_waveform[..., n_to_trim:-n_to_trim],
                    output[..., n_to_trim:-n_to_trim],
                    atol=1e-1,
                    rtol=1e-4,
                )

    def test_add_noise_broadcast(self):
        """Check that AddNoise produces correct outputs when broadcasting input dimensions."""
        leading_dims = (5, 2, 3)
        L = 51

        waveform = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device)
        noise = torch.rand(5, 1, 1, L, dtype=self.dtype, device=self.device)
        lengths = torch.rand(5, 1, 3, dtype=self.dtype, device=self.device)
        snr = torch.rand(1, 1, 1, dtype=self.dtype, device=self.device) * 10

        add_noise = T.AddNoise()
        actual = add_noise(waveform, noise, snr, lengths)

        noise_expanded = noise.expand(*leading_dims, L)
        snr_expanded = snr.expand(*leading_dims)
        lengths_expanded = lengths.expand(*leading_dims)
        expected = add_noise(waveform, noise_expanded, snr_expanded, lengths_expanded)

        self.assertEqual(expected, actual)

    @parameterized.expand(
        [((5, 2, 3), (2, 1, 1), (5, 2), (5, 2, 3)), ((2, 1), (5,), (5,), (5,)), ((3,), (5, 2, 3), (2, 1, 1), (5, 2))]
    )
    def test_add_noise_leading_dim_check(self, waveform_dims, noise_dims, lengths_dims, snr_dims):
        """Check that AddNoise properly rejects inputs with different leading dimension lengths."""
        L = 51

        waveform = torch.rand(*waveform_dims, L, dtype=self.dtype, device=self.device)
        noise = torch.rand(*noise_dims, L, dtype=self.dtype, device=self.device)
        lengths = torch.rand(*lengths_dims, dtype=self.dtype, device=self.device)
        snr = torch.rand(*snr_dims, dtype=self.dtype, device=self.device) * 10

        add_noise = T.AddNoise()

        with self.assertRaisesRegex(ValueError, "Input leading dimensions"):
            add_noise(waveform, noise, snr, lengths)

    def test_add_noise_length_check(self):
        """Check that add_noise properly rejects inputs that have inconsistent length dimensions."""
        leading_dims = (5, 2, 3)
        L = 51

        waveform = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device)
        noise = torch.rand(*leading_dims, 50, dtype=self.dtype, device=self.device)
        lengths = torch.rand(*leading_dims, dtype=self.dtype, device=self.device)
        snr = torch.rand(*leading_dims, dtype=self.dtype, device=self.device) * 10

        add_noise = T.AddNoise()

        with self.assertRaisesRegex(ValueError, "Length dimensions"):
            add_noise(waveform, noise, snr, lengths)

    @nested_params(
        [(2, 1, 31)],
        [0.97, 0.72],
    )
    def test_preemphasis(self, input_shape, coeff):
        waveform = torch.rand(*input_shape, dtype=self.dtype, device=self.device)
        preemphasis = T.Preemphasis(coeff=coeff).to(dtype=self.dtype, device=self.device)
        actual = preemphasis(waveform)

        a_coeffs = torch.tensor([1.0, 0.0], device=self.device, dtype=self.dtype)
        b_coeffs = torch.tensor([1.0, -coeff], device=self.device, dtype=self.dtype)
        expected = lfilter(waveform, a_coeffs=a_coeffs, b_coeffs=b_coeffs)
        self.assertEqual(actual, expected)

    @nested_params(
        [(2, 1, 31)],
        [0.97, 0.72],
    )
    def test_deemphasis(self, input_shape, coeff):
        waveform = torch.rand(*input_shape, dtype=self.dtype, device=self.device)
        preemphasized = preemphasis(waveform, coeff=coeff)
        deemphasis = T.Deemphasis(coeff=coeff).to(dtype=self.dtype, device=self.device)
        deemphasized = deemphasis(preemphasized)
        self.assertEqual(deemphasized, waveform)
