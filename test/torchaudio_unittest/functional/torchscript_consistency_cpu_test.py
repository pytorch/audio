import torch

from torchaudio_unittest.common_utils import PytorchTestCase
from .torchscript_consistency_impl import Functional, FunctionalComplex


class TestFunctionalFloat32(Functional, PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cpu')


class TestFunctionalFloat64(Functional, PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cpu')


class TestFunctionalComplex64(FunctionalComplex, PytorchTestCase):
    complex_dtype = torch.complex64
    real_dtype = torch.float32
    device = torch.device('cpu')


class TestFunctionalComplex128(FunctionalComplex, PytorchTestCase):
    complex_dtype = torch.complex128
    real_dtype = torch.float64
    device = torch.device('cpu')
