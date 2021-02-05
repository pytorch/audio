import torch

from torchaudio_unittest import common_utils
from .kaldi_compatibility_impl import Kaldi, KaldiCPUOnly


class TestKaldiFloat32(Kaldi, common_utils.PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cpu')


class TestKaldiFloat64(Kaldi, common_utils.PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cpu')


class TestKaldiCPUOnly(KaldiCPUOnly, common_utils.PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cpu')
