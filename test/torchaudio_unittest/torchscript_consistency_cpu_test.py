import torch

from torchaudio_unittest import common_utils
from .torchscript_consistency_impl import Transforms


class TestTransformsFloat32(Transforms, common_utils.PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cpu')


class TestTransformsFloat64(Transforms, common_utils.PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cpu')
