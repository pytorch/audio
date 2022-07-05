import torch
from torchaudio_unittest.common_utils import PytorchTestCase
from torchaudio_unittest.prototype.hdemucs_test_impl import HDemucsTests


class HDemucsFloat32CPUTest(HDemucsTests, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")
