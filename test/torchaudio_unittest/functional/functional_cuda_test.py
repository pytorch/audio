import torch
import unittest

from torchaudio_unittest import common_utils
from .functional_impl import Lfilter, Spectrogram


@common_utils.skipIfNoCuda
class TestLFilterFloat32(Lfilter, common_utils.PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cuda')

    @unittest.expectedFailure
    def test_9th_order_filter_stability(self):
        super().test_9th_order_filter_stability()


@common_utils.skipIfNoCuda
class TestLFilterFloat64(Lfilter, common_utils.PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cuda')


@common_utils.skipIfNoCuda
class TestSpectrogramFloat32(Spectrogram, common_utils.PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cuda')


@common_utils.skipIfNoCuda
class TestSpectrogramFloat64(Spectrogram, common_utils.PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cuda')
