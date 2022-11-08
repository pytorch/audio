import torch
from torchaudio_unittest.common_utils import PytorchTestCase

from .transforms_test_impl import TransformsTestImpl


class TransformsFloat32CPUTest(TransformsTestImpl, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")


class TransformsFloat64CPUTest(TransformsTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cpu")
