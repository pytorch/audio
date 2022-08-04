from .autograd_test_impl import AutogradTestImpl

import torch
from torchaudio_unittest.common_utils import PytorchTestCase


class TestAutogradCPUFloat64(AutogradTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cpu")
