import torch

from torchaudio_unittest.common_utils import PytorchTestCase
from .torchscript_consistency_impl import Transforms


class TestTransformsFloat32(Transforms, PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cpu')


class TestTransformsFloat64(Transforms, PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cpu')
