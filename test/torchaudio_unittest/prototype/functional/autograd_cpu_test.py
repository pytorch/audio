import torch
from torchaudio_unittest.common_utils import PytorchTestCase

from .autograd_test_impl import AutogradTestImpl


class TestAutogradCPUFloat64(AutogradTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cpu")
