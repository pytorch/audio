from parameterized import parameterized
import torch
from torch.autograd import gradcheck, gradgradcheck
import torchaudio.transforms as T

from torchaudio_unittest.common_utils import (
    TestBaseMixin,
    get_whitenoise,
)


class AutogradTestMixin(TestBaseMixin):
    def assert_grad(self, transform, *inputs):
        transform = transform.to(dtype=torch.float64, device=self.device)

        inputs_ = []
        for i in inputs:
            i.requires_grad = True
            inputs_.append(i.to(dtype=torch.float64, device=self.device))
        assert gradcheck(transform, inputs_)
        assert gradgradcheck(transform, inputs_)

    @parameterized.expand([
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
        transform = T.Spectrogram(**kwargs)
        waveform = get_whitenoise(sample_rate=8000, duration=0.05, n_channels=2)
        self.assert_grad(transform, waveform)

    def test_melspectrogram(self):
        sample_rate = 8000
        transform = T.MelSpectrogram(sample_rate=sample_rate)
        waveform = get_whitenoise(sample_rate=sample_rate, duration=0.05, n_channels=2)
        self.assert_grad(transform, waveform)
