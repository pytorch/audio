import torch

from torchaudio_unittest.common_utils import skipIfNoCuda, PytorchTestCase
from .torchscript_consistency_impl import MVDRTransforms


@skipIfNoCuda
class TestTransformsFloat32(MVDRTransforms, PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cuda')


@skipIfNoCuda
class TestTransformsFloat64(MVDRTransforms, PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cuda')
