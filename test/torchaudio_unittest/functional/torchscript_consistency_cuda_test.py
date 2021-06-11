import torch

from torchaudio_unittest.common_utils import skipIfNoCuda, PytorchTestCase
from .torchscript_consistency_impl import Functional


@skipIfNoCuda
class TestFunctionalFloat32(Functional, PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cuda')


@skipIfNoCuda
class TestFunctionalFloat64(Functional, PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cuda')
