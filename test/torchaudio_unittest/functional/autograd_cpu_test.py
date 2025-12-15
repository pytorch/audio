import torch
from torchaudio_unittest import common_utils

from .autograd_impl import Autograd, AutogradFloat32


class TestAutogradLfilterCPU(Autograd, common_utils.PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cpu")


class TestAutogradRNNTCPU(AutogradFloat32, common_utils.PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")
