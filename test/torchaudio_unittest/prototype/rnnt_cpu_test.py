import torch
from torchaudio_unittest.common_utils import PytorchTestCase
from torchaudio_unittest.prototype.rnnt_test_impl import ConformerRNNTTestImpl


class ConformerRNNTFloat32CPUTest(ConformerRNNTTestImpl, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")


class ConformerRNNTFloat64CPUTest(ConformerRNNTTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cpu")
