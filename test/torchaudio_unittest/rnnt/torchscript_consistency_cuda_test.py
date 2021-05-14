import torch

from torchaudio_unittest.common_utils import PytorchTestCase
from .utils import skipIfNoTransducer
from .torchscript_consistency_impl import RNNTLossTorchscript


@skipIfNoTransducer
class TestRNNTLoss(RNNTLossTorchscript, PytorchTestCase):
    device = torch.device('cuda')
