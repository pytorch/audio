import torch

from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoCuda
from .utils import skipIfNoTransducer
from .torchscript_consistency_impl import RNNTLossTorchscript


@skipIfNoTransducer
@skipIfNoCuda
class TestRNNTLoss(RNNTLossTorchscript, PytorchTestCase):
    device = torch.device('cuda')
