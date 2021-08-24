import torch

from torchaudio_unittest.common_utils import PytorchTestCase
from . transforms_test_impl import MVDRTestBase


class TransformsCPUFloat32Test(MVDRTestBase, PytorchTestCase):
    device = 'cpu'
    dtype = torch.float32


class TransformsCPUFloat64Test(MVDRTestBase, PytorchTestCase):
    device = 'cpu'
    dtype = torch.float64
