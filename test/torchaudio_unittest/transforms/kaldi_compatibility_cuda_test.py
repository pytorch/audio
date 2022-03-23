import torch
from torchaudio_unittest import common_utils

from .kaldi_compatibility_test_impl import KaldiRef


@common_utils.skipIfNoCuda
class TestKaldiRefFloat32(KaldiRef, common_utils.PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cuda")
