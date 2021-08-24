import torch

from torchaudio_unittest.common_utils import PytorchTestCase
from .torchscript_consistency_impl import MVDRTransforms


class TestTransformsFloat32(MVDRTransforms, PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cpu')


class TestTransformsFloat64(MVDRTransforms, PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cpu')
