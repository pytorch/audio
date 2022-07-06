import torch
from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoCuda
from torchaudio_unittest.prototype.hdemucs_test_impl import HDemucsTests, TestDemucsIntegration


@skipIfNoCuda
class HDemucsFloat32GPUTest(HDemucsTests, TestDemucsIntegration, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cuda")
