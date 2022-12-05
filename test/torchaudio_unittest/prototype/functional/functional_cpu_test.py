import torch
from torchaudio_unittest.common_utils import PytorchTestCase

from .functional_test_impl import Functional64OnlyTestImpl, FunctionalTestImpl


class FunctionalFloat32CPUTest(FunctionalTestImpl, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")


class FunctionalFloat64CPUTest(FunctionalTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cpu")


class FunctionalFloat64OnlyCPUTest(Functional64OnlyTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cpu")
