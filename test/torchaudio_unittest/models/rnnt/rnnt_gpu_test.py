import torch
from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoCuda
from torchaudio_unittest.models.rnnt.rnnt_test_impl import RNNTTestImpl


@skipIfNoCuda
class RNNTFloat32GPUTest(RNNTTestImpl, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cuda")


@skipIfNoCuda
class RNNTFloat64GPUTest(RNNTTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cuda")
