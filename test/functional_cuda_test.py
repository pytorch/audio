import torch

from . import common_utils
from .functional_impl import Lfilter


@common_utils.skipIfNoCuda
class TestLFilterFloat32(Lfilter, common_utils.TestCase):
    dtype = torch.float32
    device = torch.device('cuda')


@common_utils.skipIfNoCuda
class TestLFilterFloat64(Lfilter, common_utils.TestCase):
    dtype = torch.float64
    device = torch.device('cuda')
