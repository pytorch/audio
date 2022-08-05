import torch
from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoCuda

from .functional_test_impl import FunctionalTestImpl


@skipIfNoCuda
class FunctionalFloat32CUDATest(FunctionalTestImpl, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cuda")


@skipIfNoCuda
class FunctionalFloat64CUDATest(FunctionalTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cuda")
