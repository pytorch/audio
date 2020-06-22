import torch

from . import common_utils
from .torchscript_consistency_impl import Functional, Transforms


class TestFunctionalFloat32(Functional, common_utils.PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cpu')


class TestFunctionalFloat64(Functional, common_utils.PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cpu')


class TestTransformsFloat32(Transforms, common_utils.PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cpu')


class TestTransformsFloat64(Transforms, common_utils.PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cpu')
