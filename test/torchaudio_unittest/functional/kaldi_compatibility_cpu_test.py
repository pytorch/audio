import torch

from torchaudio_unittest.common_utils import PytorchTestCase
from .kaldi_compatibility_test_impl import Kaldi, KaldiCPUOnly


class TestKaldiCPUOnly(KaldiCPUOnly, PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cpu')


class TestKaldiFloat32(Kaldi, PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cpu')


class TestKaldiFloat64(Kaldi, PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cpu')
