import torch
from torchaudio_unittest.common_utils import PytorchTestCase
from torchaudio_unittest.models.conformer.conformer_test_impl import ConformerTestImpl


class ConformerFloat32CPUTest(ConformerTestImpl, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")


class ConformerFloat64CPUTest(ConformerTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cpu")
