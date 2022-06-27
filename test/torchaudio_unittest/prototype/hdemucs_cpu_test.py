import torch
from torchaudio_unittest.common_utils import PytorchTestCase
from torchaudio_unittest.prototype.hdemucs_test_impl import HDemucsEncoderTests, HDemucsDecoderTests, HDemucsTests


class HDemucsFloat32CPUTest(HDemucsEncoderTests, HDemucsDecoderTests, HDemucsTests, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")


#
# class HDemucsFloat64CPUTest(HDemucsEncoderTests, PytorchTestCase):
#     dtype = torch.float64
#     device = torch.device("cpu")
