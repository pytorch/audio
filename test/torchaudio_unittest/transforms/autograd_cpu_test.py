import torch
from torchaudio_unittest.common_utils import PytorchTestCase
from .autograd_test_impl import AutogradTestCase


class AutogradCPUTest(AutogradTestCase, PytorchTestCase):
    device = 'cpu'
    dtype = torch.float64
