import torch
from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoCuda
from torchaudio_unittest.models.rnnt_decoder.rnnt_decoder_test_impl import RNNTBeamSearchTestImpl


@skipIfNoCuda
class RNNTBeamSearchFloat32GPUTest(RNNTBeamSearchTestImpl, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cuda")


@skipIfNoCuda
class RNNTBeamSearchFloat64GPUTest(RNNTBeamSearchTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cuda")
