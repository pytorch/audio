import torch
from torchaudio_unittest.common_utils import PytorchTestCase
from torchaudio_unittest.prototype.hdemucs_test_impl import HDemucsTests, TestDemucsIntegration


class HDemucsFloat32CPUTest(HDemucsTests, TestDemucsIntegration, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")
