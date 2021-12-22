import torch
from torchaudio_unittest.common_utils import PytorchTestCase

from .transforms_test_impl import TransformsTestBase


class TransformsCPUFloat32Test(TransformsTestBase, PytorchTestCase):
    device = "cpu"
    dtype = torch.float32


class TransformsCPUFloat64Test(TransformsTestBase, PytorchTestCase):
    device = "cpu"
    dtype = torch.float64
