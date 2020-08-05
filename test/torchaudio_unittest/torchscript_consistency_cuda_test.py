import torch

from torchaudio_unittest import common_utils
from .torchscript_consistency_impl import Functional, Transforms


@common_utils.skipIfNoCuda
class TestFunctionalFloat32(Functional, common_utils.PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cuda')


@common_utils.skipIfNoCuda
class TestFunctionalFloat64(Functional, common_utils.PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cuda')


@common_utils.skipIfNoCuda
class TestTransformsFloat32(Transforms, common_utils.PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cuda')


@common_utils.skipIfNoCuda
class TestTransformsFloat64(Transforms, common_utils.PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cuda')
