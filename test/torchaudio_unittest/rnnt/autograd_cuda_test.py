import torch
from .autograd_impl import Autograd
from torchaudio_unittest import common_utils
from .utils import skipIfNoTransducer


@skipIfNoTransducer
@common_utils.skipIfNoCuda
class TestAutograd(Autograd, common_utils.PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cuda')
