import torch
from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoCuda

from .torchscript_consistency_impl import Functional, FunctionalFloat32Only


@skipIfNoCuda
class TestFunctionalFloat32(Functional, FunctionalFloat32Only, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cuda")


@skipIfNoCuda
class TestFunctionalFloat64(Functional, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cuda")
