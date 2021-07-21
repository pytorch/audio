import torch

from torchaudio_unittest.common_utils import PytorchTestCase
from .utils import skipIfNoRNNT
from .torchscript_consistency_impl import RNNTLossTorchscript


@skipIfNoRNNT
class TestRNNTLoss(RNNTLossTorchscript, PytorchTestCase):
    device = torch.device('cpu')
