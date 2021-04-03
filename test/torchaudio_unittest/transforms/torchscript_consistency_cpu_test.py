import torch

from torchaudio_unittest.common_utils import PytorchTestCase
from .torchscript_consistency_impl import Transforms, TransformsComplex


class TestTransformsFloat32(Transforms, PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cpu')


class TestTransformsFloat64(Transforms, PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cpu')


class TestTransformsComplex64(TransformsComplex, PytorchTestCase):
    complex_dtype = torch.complex64
    real_dtype = torch.float32
    device = torch.device('cpu')


class TestTransformsComplex128(TransformsComplex, PytorchTestCase):
    complex_dtype = torch.complex128
    real_dtype = torch.float64
    device = torch.device('cpu')
