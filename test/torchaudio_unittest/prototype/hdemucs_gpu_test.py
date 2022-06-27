import torch
from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoCuda
from torchaudio_unittest.prototype.hdemucs_test_impl import HDemucsEncoderTests, HDemucsDecoderTests, HDemucsTests


@skipIfNoCuda
class HDemucsFloat32GPUTest(HDemucsEncoderTests, HDemucsDecoderTests, HDemucsTests, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cuda")


# @skipIfNoCuda
# class HDemucsFloat64GPUTest(HDemucsEncoderTests, PytorchTestCase):
#     dtype = torch.float64
#     device = torch.device("cuda")
