import torch
from torchaudio_unittest.common_utils import (
    PytorchTestCase,
    skipIfNoCuda,
)
from .autograd_test_impl import AutogradTestCase


@skipIfNoCuda
class AutogradCUDATest(AutogradTestCase, PytorchTestCase):
    device = 'cuda'
    dtype = torch.float64
