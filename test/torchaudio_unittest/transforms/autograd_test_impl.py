from typing import List

from parameterized import parameterized
import torch
from torch.autograd import gradcheck, gradgradcheck
import torchaudio.transforms as T

from torchaudio_unittest.common_utils import (
    TestBaseMixin,
    get_whitenoise,
    nested_params,
)


# TODO:
# - replace T.Spectrogram
# - generalize it
# - move to common_utils
def get_spectrogram(return_complex):
    spectrogram = T.Spectrogram(return_complex=return_complex, power=None)
    waveform = get_whitenoise(sample_rate=8000, duration=0.05, n_channels=2)
    return spectrogram(waveform)


class AutogradTestMixin(TestBaseMixin):
    def assert_grad(
            self,
            transform: torch.nn.Module,
            inputs: List[torch.Tensor],
            *,
            nondet_tol: float = 0.0,
    ):
        transform = transform.to(dtype=torch.float64, device=self.device)

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

    @nested_params(
        [0.7, 0.8, 0.9, 1.0, 1.3],
        [True, False],
    )
    def test_timestretch(self, rate, test_complex):
        transform = T.TimeStretch(fixed_rate=rate)
        spectrogram = get_spectrogram(return_complex=test_complex)
        self.assert_grad(transform, [spectrogram])

    @nested_params(
        [0.7, 0.8, 0.9, 1.0, 1.3],
        [True, False],
    )
    def test_timestretch_override(self, rate, test_complex):
        transform = T.TimeStretch()
        spectrogram = get_spectrogram(return_complex=test_complex)
        self.assert_grad(transform, [spectrogram, rate])
