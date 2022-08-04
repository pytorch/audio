from .autograd_test_impl import AutogradTestImpl

import torch
from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoCuda


@skipIfNoCuda
class TestAutogradCUDAFloat64(AutogradTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cuda")
