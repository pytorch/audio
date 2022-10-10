import torch
from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoCuda
from torchaudio_unittest.models.hdemucs.hdemucs_test_impl import CompareHDemucsOriginal, HDemucsTests


@skipIfNoCuda
class HDemucsFloat32GPUTest(HDemucsTests, CompareHDemucsOriginal, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cuda")
