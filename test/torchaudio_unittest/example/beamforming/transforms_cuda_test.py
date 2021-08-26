import torch

from torchaudio_unittest.common_utils import (
    PytorchTestCase,
    skipIfNoCuda,
)
from . transforms_test_impl import TransformsTestBase


@skipIfNoCuda
class TransformsCPUFloat32Test(TransformsTestBase, PytorchTestCase):
    device = 'cuda'
    dtype = torch.float32


@skipIfNoCuda
class TransformsCPUFloat64Test(TransformsTestBase, PytorchTestCase):
    device = 'cpu'
    dtype = torch.float64
