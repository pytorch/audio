import torch
from torchaudio_unittest import common_utils

from .kaldi_compatibility_test_impl import Kaldi


@common_utils.skipIfNoCuda
class TestKaldiFloat32(Kaldi, common_utils.PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cuda")
