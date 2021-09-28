import torch
from torchaudio_unittest.prototype.emformer_test_impl import EmformerTestImpl
from torchaudio_unittest.common_utils import PytorchTestCase


class EmformerFloat32CPUTest(EmformerTestImpl, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")


class EmformerFloat64CPUTest(EmformerTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cpu")
