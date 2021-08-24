import torch

from torchaudio_unittest.common_utils import (
    PytorchTestCase,
    skipIfNoCuda,
)
from . transforms_test_impl import MVDRTestBase


@skipIfNoCuda
class TransformsCPUFloat32Test(MVDRTestBase, PytorchTestCase):
    device = 'cuda'
    dtype = torch.float32


@skipIfNoCuda
class TransformsCPUFloat64Test(MVDRTestBase, PytorchTestCase):
    device = 'cpu'
    dtype = torch.float64
