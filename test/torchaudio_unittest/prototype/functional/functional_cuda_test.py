import torch
from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoCuda

from .functional_test_impl import Functional64OnlyTestImpl, FunctionalTestImpl


@skipIfNoCuda
class FunctionalFloat32CUDATest(FunctionalTestImpl, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cuda", 0)


@skipIfNoCuda
class FunctionalFloat64CUDATest(FunctionalTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cuda", 0)


@skipIfNoCuda
class FunctionalFloat64OnlyCUDATest(Functional64OnlyTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cuda")
