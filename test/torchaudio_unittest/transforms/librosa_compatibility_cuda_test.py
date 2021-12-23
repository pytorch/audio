import torch
from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoCuda

from .librosa_compatibility_test_impl import TransformsTestBase


@skipIfNoCuda
class TestTransforms(TransformsTestBase, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cuda")
