from typing import List

from parameterized import parameterized, param
import torch
from beamforming.mvdr import PSD, MVDR
from torch.autograd import gradcheck, gradgradcheck

from torchaudio_unittest.common_utils import (
    TestBaseMixin,
    get_whitenoise,
    get_spectrogram,
)


class AutogradTestMixin(TestBaseMixin):
    def assert_grad(
            self,
            transform: torch.nn.Module,
            inputs: List[torch.Tensor],
            *,
            nondet_tol: float = 0.0,
    ):
        transform = transform.to(dtype=torch.float64, device=self.device)

        # gradcheck and gradgradcheck only pass if the input tensors are of dtype `torch.double` or
        # `torch.cdouble`, when the default eps and tolerance values are used.
        inputs_ = []
        for i in inputs:
            if torch.is_tensor(i):
                i = i.to(
                    dtype=torch.cdouble if i.is_complex() else torch.double,
                    device=self.device)
                i.requires_grad = True
            inputs_.append(i)
        assert gradcheck(transform, inputs_)
        assert gradgradcheck(transform, inputs_, nondet_tol=nondet_tol)

    def test_psd(self):
        transform = PSD()
        waveform = get_whitenoise(sample_rate=8000, duration=0.05, n_channels=2)
        spectrogram = get_spectrogram(waveform, n_fft=400)
        self.assert_grad(transform, [spectrogram])

    @parameterized.expand([
        [True],
        [False],
    ])
    def test_psd_with_mask(self, multi_mask):
        transform = PSD(multi_mask=multi_mask)
        waveform = get_whitenoise(sample_rate=8000, duration=0.05, n_channels=2)
        spectrogram = get_spectrogram(waveform, n_fft=400)
        if multi_mask:
            mask = torch.rand(spectrogram.shape[-3:])
        else:
            mask = torch.rand(spectrogram.shape[-2:])

        self.assert_grad(transform, [spectrogram, mask])

    @parameterized.expand([
        param(solution="ref_channel"),
        param(solution="stv_power"),
        # evd will fail since the eigenvalues are not distinct
        # param(solution="stv_evd"),
    ])
    def test_mvdr(self, solution):
        transform = MVDR(solution=solution)
        waveform = get_whitenoise(sample_rate=8000, duration=0.05, n_channels=2)
        spectrogram = get_spectrogram(waveform, n_fft=400)
        mask = torch.rand(spectrogram.shape[-2:])
        self.assert_grad(transform, [spectrogram, mask])
