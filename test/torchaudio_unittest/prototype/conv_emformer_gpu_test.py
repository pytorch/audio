import torch
from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoCuda
from torchaudio_unittest.prototype.conv_emformer_test_impl import ConvEmformerTestImpl


@skipIfNoCuda
class ConvEmformerFloat32GPUTest(ConvEmformerTestImpl, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cuda")


@skipIfNoCuda
class ConvEmformerFloat64GPUTest(ConvEmformerTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cuda")
