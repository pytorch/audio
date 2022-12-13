import torch
from torchaudio_unittest.common_utils import PytorchTestCase

from .hifi_gan_test_impl import HiFiGANTestImpl


class HiFiGANFloat32CPUTest(HiFiGANTestImpl, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")


class HiFiGANFloat64CPUTest(HiFiGANTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cpu")
