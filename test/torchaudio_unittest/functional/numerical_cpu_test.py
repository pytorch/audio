import torch
from .numerical_test_impl import NumericalStability
from torchaudio_unittest import common_utils


class TestNumericalStabilityCPU64(NumericalStability, common_utils.PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cpu')


class TestNumericalStabilityCPU32(NumericalStability, common_utils.PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cpu')
