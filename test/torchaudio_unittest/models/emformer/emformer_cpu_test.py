import torch
from torchaudio_unittest.common_utils import PytorchTestCase
from torchaudio_unittest.models.emformer.emformer_test_impl import EmformerTestImpl


class EmformerFloat32CPUTest(EmformerTestImpl, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")


class EmformerFloat64CPUTest(EmformerTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cpu")
