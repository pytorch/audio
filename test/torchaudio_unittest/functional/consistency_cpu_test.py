import torch

from torchaudio_unittest import common_utils
from .consistency_impl import Functional


class TestFunctionalFloat32(Functional, common_utils.PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cpu')


class TestFunctionalFloat64(Functional, common_utils.PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cpu')
