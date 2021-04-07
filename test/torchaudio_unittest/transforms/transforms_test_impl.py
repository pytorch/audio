import torch
import torchaudio.transforms as T

from torchaudio_unittest.common_utils import (
    TestBaseMixin,
    get_whitenoise,
    get_spectrogram,
)


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
            get_whitenoise(sample_rate=sample_rate, duration=1, n_channels=2),
            n_fft=n_fft, power=power).to(self.device, self.dtype)
        input = T.MelScale(
            n_mels=n_mels, sample_rate=sample_rate
        ).to(self.device, self.dtype)(expected)

        # Run transform
        transform = T.InverseMelScale(
            n_stft, n_mels=n_mels, sample_rate=sample_rate).to(self.device, self.dtype)
        torch.random.manual_seed(0)
        result = transform(input)

        # Compare
        epsilon = 1e-60
        relative_diff = torch.abs((result - expected) / (expected + epsilon))

        for tol in [1e-1, 1e-3, 1e-5, 1e-10]:
            print(
                f"Ratio of relative diff smaller than {tol:e} is "
                f"{_get_ratio(relative_diff < tol)}")
        assert _get_ratio(relative_diff < 1e-1) > 0.2
        assert _get_ratio(relative_diff < 1e-3) > 5e-3
        assert _get_ratio(relative_diff < 1e-5) > 1e-5
