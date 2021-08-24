from parameterized import parameterized, param
from typing import Optional
from beamforming.mvdr import PSD
import numpy as np
import torch
from torchaudio_unittest.common_utils import (
    TestBaseMixin,
    get_whitenoise,
    get_spectrogram,
)


def psd_numpy(
        X: np.array,
        mask: Optional[np.array],
        normalize: bool = True,
        eps: float = 1e-15
) -> np.array:
    X_conj = np.conj(X)
    psd_X = np.einsum("...ct,...et->...tce", X, X_conj)
    if mask is not None:
        if normalize:
            mask = mask / (mask.sum(axis=-1, keepdims=True) + eps)
        psd = psd_X * mask[..., None, None]
    else:
        psd = psd_X

    psd = psd.sum(axis=-3)

    return torch.tensor(psd, dtype=torch.cdouble)


class MVDRTestBase(TestBaseMixin):
    @parameterized.expand([
        param(0.5, 1, True),
        param(0.5, 1, None),
        param(1, 4, True),
        param(1, 6, None),
    ])
    def test_psd(self, duration, channel, mask):
        """Providing dtype changes the kernel cache dtype"""
        transform = PSD()
        waveform = get_whitenoise(sample_rate=8000, duration=duration, n_channels=channel)
        spectrogram = get_spectrogram(waveform, n_fft=400)  # (channel, freq, time)
        spectrogram = spectrogram.to(torch.cdouble)
        spectrogram = spectrogram.transpose(-2, -3)
        if mask is not None:
            mask = torch.rand(spectrogram.shape[-3], spectrogram.shape[-1])
            psd_np = psd_numpy(spectrogram.detach().numpy(), mask.detach().numpy())
        else:
            psd_np = psd_numpy(spectrogram.detach().numpy(), mask)
        psd = transform(spectrogram, mask)
        self.assertEqual(psd, psd_np, atol=1e-5, rtol=1e-5)
