import torch
import torchaudio.prototype.transforms as T
from torchaudio_unittest.common_utils import get_spectrogram, get_whitenoise, TestBaseMixin


def _get_ratio(mat):
    return (mat.sum() / mat.numel()).item()


class TransformsTestImpl(TestBaseMixin):
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
