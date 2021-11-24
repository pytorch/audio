import torch
from torchaudio_unittest.prototype.rnnt_decoder_test_impl import RNNTBeamSearchTestImpl
from torchaudio_unittest.common_utils import PytorchTestCase


class RNNTBeamSearchFloat32CPUTest(RNNTBeamSearchTestImpl, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")


class RNNTBeamSearchFloat64CPUTest(RNNTBeamSearchTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cpu")
