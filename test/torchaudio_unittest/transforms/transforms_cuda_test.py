import torch
from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoCuda

from .transforms_test_impl import TransformsTestBase


@skipIfNoCuda
class TransformsCUDAFloat32Test(TransformsTestBase, PytorchTestCase):
    device = "cuda"
    dtype = torch.float32


@skipIfNoCuda
class TransformsCUDAFloat64Test(TransformsTestBase, PytorchTestCase):
    device = "cuda"
    dtype = torch.float64
