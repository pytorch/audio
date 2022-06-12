import torch
from torchaudio_unittest import common_utils

from .loudness_compliance_test_impl import Loudness


class TestLoudnessCPU(Loudness, common_utils.PytorchTestCase):
    device = torch.device("cpu")
