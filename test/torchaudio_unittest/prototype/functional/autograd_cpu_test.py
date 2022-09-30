import torch
from torchaudio_unittest.common_utils import PytorchTestCase

from .autograd_test_impl import AutogradTestImpl, AutogradTestRIRImpl


class TestAutogradCPUFloat64(AutogradTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cpu")


class TestAutogradRIRCPUFloat64(AutogradTestRIRImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cpu")
