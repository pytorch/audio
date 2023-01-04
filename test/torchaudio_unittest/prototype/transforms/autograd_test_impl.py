from typing import List

import torch
import torchaudio.prototype.transforms as T
from torch.autograd import gradcheck, gradgradcheck
from torchaudio_unittest.common_utils import get_spectrogram, get_whitenoise, nested_params, TestBaseMixin


class Autograd(TestBaseMixin):
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
                i = i.to(dtype=torch.cdouble if i.is_complex() else torch.double, device=self.device)
                i.requires_grad = True
            inputs_.append(i)
        assert gradcheck(transform, inputs_)
        assert gradgradcheck(transform, inputs_, nondet_tol=nondet_tol)

    @nested_params(
        [T.Convolve, T.FFTConvolve],
        ["full", "valid", "same"],
    )
    def test_Convolve(self, cls, mode):
        leading_dims = (4, 3, 2)
        L_x, L_y = 23, 40
        x = torch.rand(*leading_dims, L_x, dtype=self.dtype, device=self.device)
        y = torch.rand(*leading_dims, L_y, dtype=self.dtype, device=self.device)
        convolve = cls(mode=mode).to(dtype=self.dtype, device=self.device)
        self.assert_grad(convolve, [x, y])

    def test_barkspectrogram(self):
        # replication_pad1d_backward_cuda is not deteministic and
        # gives very small (~e-16) difference.
        sample_rate = 8000
        transform = T.BarkSpectrogram(sample_rate=sample_rate)
        waveform = get_whitenoise(sample_rate=sample_rate, duration=0.05, n_channels=2)
        self.assert_grad(transform, [waveform], nondet_tol=1e-10)

    def test_barkscale(self):
        sample_rate = 8000
        n_fft = 400
        n_barks = n_fft // 2 + 1
        transform = T.BarkScale(sample_rate=sample_rate, n_barks=n_barks)
        spec = get_spectrogram(
            get_whitenoise(sample_rate=sample_rate, duration=0.05, n_channels=2), n_fft=n_fft, power=1
        )
        self.assert_grad(transform, [spec])

    def test_Speed(self):
        leading_dims = (3, 2)
        time = 200
        waveform = torch.rand(*leading_dims, time, dtype=torch.float64, device=self.device, requires_grad=True)
        lengths = torch.randint(1, time, leading_dims, dtype=torch.float64, device=self.device)
        speed = T.Speed(1000, 1.1).to(device=self.device, dtype=torch.float64)
        assert gradcheck(speed, (waveform, lengths))
        assert gradgradcheck(speed, (waveform, lengths))

    def test_SpeedPerturbation(self):
        leading_dims = (3, 2)
        time = 200
        waveform = torch.rand(*leading_dims, time, dtype=torch.float64, device=self.device, requires_grad=True)
        lengths = torch.randint(1, time, leading_dims, dtype=torch.float64, device=self.device)
        speed = T.SpeedPerturbation(1000, [0.9]).to(device=self.device, dtype=torch.float64)
        assert gradcheck(speed, (waveform, lengths))
        assert gradgradcheck(speed, (waveform, lengths))

    def test_AddNoise(self):
        leading_dims = (2, 3)
        L = 31

        waveform = torch.rand(*leading_dims, L, dtype=torch.float64, device=self.device, requires_grad=True)
        noise = torch.rand(*leading_dims, L, dtype=torch.float64, device=self.device, requires_grad=True)
        lengths = torch.rand(*leading_dims, dtype=torch.float64, device=self.device, requires_grad=True)
        snr = torch.rand(*leading_dims, dtype=torch.float64, device=self.device, requires_grad=True) * 10

        add_noise = T.AddNoise().to(self.device, torch.float64)
        assert gradcheck(add_noise, (waveform, noise, lengths, snr))
        assert gradgradcheck(add_noise, (waveform, noise, lengths, snr))

    def test_Preemphasis(self):
        waveform = torch.rand(3, 4, 10, dtype=torch.float64, device=self.device, requires_grad=True)
        preemphasis = T.Preemphasis(coeff=0.97).to(dtype=torch.float64, device=self.device)
        assert gradcheck(preemphasis, (waveform,))
        assert gradgradcheck(preemphasis, (waveform,))

    def test_Deemphasis(self):
        waveform = torch.rand(3, 4, 10, dtype=torch.float64, device=self.device, requires_grad=True)
        deemphasis = T.Deemphasis(coeff=0.97).to(dtype=torch.float64, device=self.device)
        assert gradcheck(deemphasis, (waveform,))
        assert gradgradcheck(deemphasis, (waveform,))
