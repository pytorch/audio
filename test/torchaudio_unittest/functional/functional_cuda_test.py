import unittest

import torch
from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoCuda, skipIfSingleCuda

from .functional_impl import Functional, FunctionalCUDAOnly


@skipIfNoCuda
class TestFunctionalFloat32(Functional, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cuda")

    @unittest.expectedFailure
    def test_lfilter_9th_order_filter_stability(self):
        super().test_lfilter_9th_order_filter_stability()


@skipIfNoCuda
class TestLFilterFloat64(Functional, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cuda")


@skipIfNoCuda
class TestFunctionalCUDAOnlyFloat32(FunctionalCUDAOnly, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cuda")


@skipIfSingleCuda
class TestFunctionalMultiGPUCUDAOnlyFloat32(FunctionalCUDAOnly, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cuda:1")


@skipIfNoCuda
class TestFunctionalCUDAOnlyFloat64(FunctionalCUDAOnly, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cuda")
