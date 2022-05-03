import torch
from torchaudio_unittest.common_utils import PytorchTestCase
from torchaudio_unittest.prototype.conv_emformer_test_impl import ConvEmformerTestImpl


class ConvEmformerFloat32CPUTest(ConvEmformerTestImpl, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")


class ConvEmformerFloat64CPUTest(ConvEmformerTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cpu")
