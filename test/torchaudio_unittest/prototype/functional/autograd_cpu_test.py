import torch
from torchaudio_unittest.common_utils import PytorchTestCase

from .autograd_test_impl import AutogradTestImpl, AutogradTestRayTracingImpl


class TestAutogradCPUFloat64(AutogradTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cpu")


class TestAutogradRayTracingCPUFloat64(AutogradTestRayTracingImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cpu")
