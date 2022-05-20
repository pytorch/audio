import torch
from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoCuda

from .torchscript_consistency_impl import Transforms, TransformsFloat32Only


@skipIfNoCuda
class TestTransformsFloat32(Transforms, TransformsFloat32Only, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cuda")


@skipIfNoCuda
class TestTransformsFloat64(Transforms, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cuda")
