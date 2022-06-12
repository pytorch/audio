import torch
from torchaudio_unittest import common_utils

from .loudness_compliance_test_impl import Loudness


@common_utils.skipIfNoCuda
class TestLoudnessCUDA(Loudness, common_utils.PytorchTestCase):
    device = torch.device("cuda")
