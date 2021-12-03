import torch
from .autograd_impl import Autograd, AutogradFloat32
from torchaudio_unittest import common_utils


class TestAutogradLfilterCPU(Autograd, common_utils.PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cpu')


class TestAutogradRNNTCPU(AutogradFloat32, common_utils.PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cpu')
