from typing import List
import unittest

from parameterized import parameterized
import torch
from torch.autograd import gradcheck, gradgradcheck
import torchaudio.transforms as T

from torchaudio_unittest.common_utils import (
    TestBaseMixin,
    get_whitenoise,
    get_spectrogram,
    nested_params,
)


class _DeterministicWrapper(torch.nn.Module):
    """Helper transform wrapper to make the given transform deterministic"""
    def __init__(self, transform, seed=0):
        super().__init__()
        self.seed = seed
        self.transform = transform

    def forward(self, input: torch.Tensor):
        torch.random.manual_seed(self.seed)
        return self.transform(input)


class AutogradTestMixin(TestBaseMixin):
    def assert_grad(
            self,
            transform: torch.nn.Module,
            inputs: List[torch.Tensor],
            *,
            nondet_tol: float = 0.0,
    ):
        transform = transform.to(dtype=torch.float64, device=self.device)

        # For the autograd test, float32 and cfloat64 are not good enough
        # we make sure that all the tensors are of float64 or cfloat128
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

    @parameterized.expand([
        ({'pad': 0, 'normalized': False, 'power': None, 'return_complex': True}, ),
        ({'pad': 3, 'normalized': False, 'power': None, 'return_complex': True}, ),
        ({'pad': 0, 'normalized': True, 'power': None, 'return_complex': True}, ),
        ({'pad': 3, 'normalized': True, 'power': None, 'return_complex': True}, ),
        ({'pad': 0, 'normalized': False, 'power': None}, ),
        ({'pad': 3, 'normalized': False, 'power': None}, ),
        ({'pad': 0, 'normalized': True, 'power': None}, ),
        ({'pad': 3, 'normalized': True, 'power': None}, ),
        ({'pad': 0, 'normalized': False, 'power': 1.0}, ),
        ({'pad': 3, 'normalized': False, 'power': 1.0}, ),
        ({'pad': 0, 'normalized': True, 'power': 1.0}, ),
        ({'pad': 3, 'normalized': True, 'power': 1.0}, ),
        ({'pad': 0, 'normalized': False, 'power': 2.0}, ),
        ({'pad': 3, 'normalized': False, 'power': 2.0}, ),
        ({'pad': 0, 'normalized': True, 'power': 2.0}, ),
        ({'pad': 3, 'normalized': True, 'power': 2.0}, ),
    ])
    def test_spectrogram(self, kwargs):
        # replication_pad1d_backward_cuda is not deteministic and
        # gives very small (~2.7756e-17) difference.
        #
        # See https://github.com/pytorch/pytorch/issues/54093
        transform = T.Spectrogram(**kwargs)
        waveform = get_whitenoise(sample_rate=8000, duration=0.05, n_channels=2)
        self.assert_grad(transform, [waveform], nondet_tol=1e-10)

    def test_melspectrogram(self):
        # replication_pad1d_backward_cuda is not deteministic and
        # gives very small (~2.7756e-17) difference.
        #
        # See https://github.com/pytorch/pytorch/issues/54093
        sample_rate = 8000
        transform = T.MelSpectrogram(sample_rate=sample_rate)
        waveform = get_whitenoise(sample_rate=sample_rate, duration=0.05, n_channels=2)
        self.assert_grad(transform, [waveform], nondet_tol=1e-10)

    @nested_params(
        [0, 0.99],
        [False, True],
    )
    def test_griffinlim(self, momentum, rand_init):
        n_fft = 400
        power = 1
        n_iter = 3
        spec = get_spectrogram(
            get_whitenoise(sample_rate=8000, duration=0.05, n_channels=2),
            n_fft=n_fft, power=power)
        transform = _DeterministicWrapper(
            T.GriffinLim(n_fft=n_fft, n_iter=n_iter, momentum=momentum, rand_init=rand_init, power=power))
        self.assert_grad(transform, [spec])

    @parameterized.expand([(False, ), (True, )])
    def test_mfcc(self, log_mels):
        sample_rate = 8000
        transform = T.MFCC(sample_rate=sample_rate, log_mels=log_mels)
        waveform = get_whitenoise(sample_rate=sample_rate, duration=0.05, n_channels=2)
        self.assert_grad(transform, [waveform])

    def test_compute_deltas(self):
        transform = T.ComputeDeltas()
        spec = torch.rand(10, 20)
        self.assert_grad(transform, [spec])

    @parameterized.expand([(8000, 8000), (8000, 4000), (4000, 8000)])
    def test_resample(self, orig_freq, new_freq):
        transform = T.Resample(orig_freq=orig_freq, new_freq=new_freq)
        waveform = get_whitenoise(sample_rate=8000, duration=0.05, n_channels=2)
        self.assert_grad(transform, [waveform])

    @parameterized.expand([("linear", ), ("exponential", ), ("logarithmic", ), ("quarter_sine", ), ("half_sine", )])
    def test_fade(self, fade_shape):
        transform = T.Fade(fade_shape=fade_shape)
        waveform = get_whitenoise(sample_rate=8000, duration=0.05, n_channels=2)
        self.assert_grad(transform, [waveform], nondet_tol=1e-10)

    def test_spectral_centroid(self):
        sample_rate = 8000
        transform = T.SpectralCentroid(sample_rate=sample_rate)
        waveform = get_whitenoise(sample_rate=sample_rate, duration=0.05, n_channels=2)
        self.assert_grad(transform, [waveform], nondet_tol=1e-10)

    def test_amplitude_to_db(self):
        sample_rate = 8000
        transform = T.AmplitudeToDB()
        waveform = get_whitenoise(sample_rate=sample_rate, duration=0.05, n_channels=2)
        self.assert_grad(transform, [waveform])

    @unittest.expectedFailure
    def test_timestretch_zeros_fail(self):
        """Test that ``T.TimeStretch`` fails gradcheck at 0

        This is because ``F.phase_vocoder`` converts data from cartesian to polar coordinate,
        which performs ``atan2(img, real)``, and gradient is not defined at 0.
        """
        n_fft = 16
        transform = T.TimeStretch(n_freq=n_fft // 2 + 1, fixed_rate=0.99)
        waveform = torch.zeros(2, 40)
        spectrogram = get_spectrogram(waveform, n_fft=n_fft, power=None)
        self.assert_grad(transform, [spectrogram])

    @nested_params(
        [0.7, 0.8, 0.9, 1.0, 1.3],
        [False, True],
    )
    def test_timestretch(self, rate, test_pseudo_complex):
        """Verify that ``T.TimeStretch`` does not fail if it's not close to 0

        ``T.TimeStrech`` is not differentiable around 0, so this test checks the differentiability
        for cases where input is not zero.

        As tested above, when spectrogram contains values close to zero, the gradients are unstable
        and gradcheck fails.

        In this test, we generate spectrogram from random signal, then we push the points around
        zero away from the origin.

        This process does not reflect the real use-case, and it is not practical for users, but
        this helps us understand to what degree the function is differentiable and when not.
        """
        n_fft = 16
        transform = T.TimeStretch(n_freq=n_fft // 2 + 1, fixed_rate=rate)
        waveform = get_whitenoise(sample_rate=40, duration=1, n_channels=2)
        spectrogram = get_spectrogram(waveform, n_fft=n_fft, power=None)

        # 1e-3 is too small (on CPU)
        epsilon = 1e-2
        too_close = spectrogram.abs() < epsilon
        spectrogram[too_close] = epsilon * spectrogram[too_close] / spectrogram[too_close].abs()
        if test_pseudo_complex:
            spectrogram = torch.view_as_real(spectrogram)
        self.assert_grad(transform, [spectrogram])
