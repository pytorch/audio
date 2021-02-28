import torch

from torchaudio_unittest.common_utils import PytorchTestCase
from .kaldi_compatibility_test_impl import KaldiCPUOnly, Kaldi


class TestKaldiCPUOnly(KaldiCPUOnly, PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cpu')


class TestKaldiCPUOnly(Kaldi, PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cpu')
