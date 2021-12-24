import torch
from torchaudio_unittest.common_utils import skipIfNoCuda, PytorchTestCase
from torchaudio_unittest.prototype.rnnt_decoder_test_impl import RNNTBeamSearchTestImpl


@skipIfNoCuda
class RNNTBeamSearchFloat32GPUTest(RNNTBeamSearchTestImpl, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cuda")


@skipIfNoCuda
class RNNTBeamSearchFloat64GPUTest(RNNTBeamSearchTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cuda")
