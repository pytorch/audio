import torch

from torchaudio_unittest import common_utils
from .torchscript_consistency_impl import Functional, Transforms, TransformsWithComplexDtypes


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


@common_utils.skipIfNoCuda
class TestTransformsCFloat(TransformsWithComplexDtypes, common_utils.PytorchTestCase):
    dtype = torch.cfloat
    device = torch.device('cuda')


@common_utils.skipIfNoCuda
class TestTransformsCDouble(TransformsWithComplexDtypes, common_utils.PytorchTestCase):
    dtype = torch.cdouble
    device = torch.device('cuda')
