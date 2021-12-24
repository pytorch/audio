import torch
from torchaudio_unittest.common_utils import PytorchTestCase

from .librosa_compatibility_test_impl import TransformsTestBase


class TestTransforms(TransformsTestBase, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cpu")
