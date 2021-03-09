import torch
from .numerical_test_impl import NumericalStability
from torchaudio_unittest import common_utils


@common_utils.skipIfNoCuda
class TestNumericalStabilityCUDA64(NumericalStability, common_utils.PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cuda')


@common_utils.skipIfNoCuda
class TestNumericalStabilityCUDA32(NumericalStability, common_utils.PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cuda')
