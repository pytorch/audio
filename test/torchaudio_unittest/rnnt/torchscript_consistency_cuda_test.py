import torch

from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoCuda
from .utils import skipIfNoRNNT
from .torchscript_consistency_impl import RNNTLossTorchscript


@skipIfNoRNNT
@skipIfNoCuda
class TestRNNTLoss(RNNTLossTorchscript, PytorchTestCase):
    device = torch.device('cuda')
