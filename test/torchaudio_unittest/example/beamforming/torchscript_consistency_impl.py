"""Test suites for jit-ability and its numerical compatibility"""

import torch
from beamforming.mvdr import PSD, MVDR
from parameterized import parameterized, param

from torchaudio_unittest import common_utils
from torchaudio_unittest.common_utils import (
    TempDirMixin,
    TestBaseMixin,
)


class Transforms(TempDirMixin, TestBaseMixin):
    """Implements test for Transforms that are performed for different devices"""
    def _assert_consistency_complex(self, transform, tensors):
        assert tensors[0].is_complex()
        tensors = [tensor.to(device=self.device, dtype=self.complex_dtype) for tensor in tensors]
        transform = transform.to(device=self.device, dtype=self.dtype)

        path = self.get_temp_path('func.zip')
        torch.jit.script(transform).save(path)
        ts_transform = torch.jit.load(path)

        output = transform(*tensors)
        ts_output = ts_transform(*tensors)
        self.assertEqual(ts_output, output)

    def test_PSD(self):
        tensor = common_utils.get_whitenoise(sample_rate=8000, n_channels=4)
        spectrogram = common_utils.get_spectrogram(tensor, n_fft=400, hop_length=100)
        self._assert_consistency_complex(PSD(), (spectrogram,))

    def test_PSD_with_mask(self):
        tensor = common_utils.get_whitenoise(sample_rate=8000, n_channels=4)
        spectrogram = common_utils.get_spectrogram(tensor, n_fft=400, hop_length=100)
        mask = torch.rand(spectrogram.shape[-2:])
        self._assert_consistency_complex(PSD(), (spectrogram, mask))


class TransformsFloat64Only(TestBaseMixin):
    @parameterized.expand([
        param(solution="ref_channel"),
        param(solution="stv_evd"),
        param(solution="stv_power"),
    ])
    def test_MVDR(self, solution):
        tensor = common_utils.get_whitenoise(sample_rate=8000, n_channels=4)
        spectrogram = common_utils.get_spectrogram(tensor, n_fft=400, hop_length=100)
        mask = torch.rand(spectrogram.shape[-2:])
        self._assert_consistency_complex(MVDR(solution=solution), (spectrogram, mask))
