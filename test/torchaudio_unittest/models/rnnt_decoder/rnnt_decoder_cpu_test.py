import torch
from torchaudio_unittest.common_utils import PytorchTestCase
from torchaudio_unittest.models.rnnt_decoder.rnnt_decoder_test_impl import RNNTBeamSearchTestImpl


class RNNTBeamSearchFloat32CPUTest(RNNTBeamSearchTestImpl, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")


class RNNTBeamSearchFloat64CPUTest(RNNTBeamSearchTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cpu")
