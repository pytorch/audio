import torch
import torchaudio.transforms as T
from parameterized import param, parameterized
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

    def test_InverseBarkScale(self):
        """Gauge the quality of InverseBarkScale transform.

        As InverseBarkScale is currently implemented with
        random initialization + iterative optimization,
        it is not practically possible to assert the difference between
        the estimated spectrogram and the original spectrogram as a whole.
        Estimated spectrogram has very huge descrepency locally.
        Thus in this test we gauge what percentage of elements are bellow
        certain tolerance.
        At the moment, the quality of estimated spectrogram is worse than the
        one obtained for Inverse MelScale.
        When implementation is changed in a way it makes the quality even worse,
        this test will fail.
        """
        n_fft = 400
        power = 1
        n_barks = 64
        sample_rate = 8000

        n_stft = n_fft // 2 + 1

        # Generate reference spectrogram and input mel-scaled spectrogram
        expected = get_spectrogram(
            get_whitenoise(sample_rate=sample_rate, duration=1, n_channels=2), n_fft=n_fft, power=power
        ).to(self.device, self.dtype)
        input = T.BarkScale(n_barks=n_barks, sample_rate=sample_rate, n_stft=n_stft).to(self.device, self.dtype)(
            expected
        )

        # Run transform
        transform = T.InverseBarkScale(n_stft, n_barks=n_barks, sample_rate=sample_rate).to(self.device, self.dtype)
        result = transform(input)

        # Compare
        epsilon = 1e-60
        relative_diff = torch.abs((result - expected) / (expected + epsilon))

        for tol in [1e-1, 1e-3, 1e-5, 1e-10]:
            print(f"Ratio of relative diff smaller than {tol:e} is " f"{_get_ratio(relative_diff < tol)}")
        assert _get_ratio(relative_diff < 1e-1) > 0.2
        assert _get_ratio(relative_diff < 1e-3) > 2e-3

    @nested_params(
        ["sinc_interpolation", "kaiser_window"],
        [16000, 44100],
    )
    def test_resample_identity(self, resampling_method, sample_rate):
        """When sampling rate is not changed, the transform returns an identical Tensor"""
        waveform = get_whitenoise(sample_rate=sample_rate, duration=1)

        resampler = T.Resample(sample_rate, sample_rate, resampling_method)
        resampled = resampler(waveform)
        self.assertEqual(waveform, resampled)

    @nested_params(
        ["sinc_interpolation", "kaiser_window"],
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
