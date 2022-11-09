import torch
from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoCuda

from .transforms_test_impl import TransformsTestImpl


@skipIfNoCuda
class TransformsFloat32CUDATest(TransformsTestImpl, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cuda")


@skipIfNoCuda
class TransformsFloat64CUDATest(TransformsTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cuda")
