import torch
import unittest

from torchaudio_unittest import common_utils
from .functional_impl import Functional, FunctionalComplex


@common_utils.skipIfNoCuda
class TestFunctionalloat32(Functional, common_utils.PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cuda')

    @unittest.expectedFailure
    def test_lfilter_9th_order_filter_stability(self):
        super().test_lfilter_9th_order_filter_stability()


@common_utils.skipIfNoCuda
class TestLFilterFloat64(Functional, common_utils.PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cuda')


@common_utils.skipIfNoCuda
class TestFunctionalComplex64(FunctionalComplex, common_utils.PytorchTestCase):
    complex_dtype = torch.complex64
    real_dtype = torch.float32
    device = torch.device('cuda')


@common_utils.skipIfNoCuda
class TestFunctionalComplex128(FunctionalComplex, common_utils.PytorchTestCase):
    complex_dtype = torch.complex64
    real_dtype = torch.float32
    device = torch.device('cuda')
