import torch
import unittest

from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoCuda
from .functional_impl import Functional, FunctionalComplex


@skipIfNoCuda
class TestFunctionalFloat32(Functional, PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cuda')

    @unittest.expectedFailure
    def test_lfilter_9th_order_filter_stability(self):
        super().test_lfilter_9th_order_filter_stability()


@skipIfNoCuda
class TestLFilterFloat64(Functional, PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cuda')


@skipIfNoCuda
class TestFunctionalComplex64(FunctionalComplex, PytorchTestCase):
    complex_dtype = torch.complex64
    real_dtype = torch.float32
    device = torch.device('cuda')


@skipIfNoCuda
class TestFunctionalComplex128(FunctionalComplex, PytorchTestCase):
    complex_dtype = torch.complex64
    real_dtype = torch.float32
    device = torch.device('cuda')
