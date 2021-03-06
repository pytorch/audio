import torch
from .autograd_impl import Autograd
from torchaudio_unittest import common_utils


class TestAutogradLfilterCPU(Autograd, common_utils.PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cpu')
