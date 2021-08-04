import torch
from torchaudio_unittest import common_utils
from .utils import skipIfNoRNNT
from .rnnt_loss_impl import RNNTLossTest


@skipIfNoRNNT
class TestRNNTLoss(RNNTLossTest, common_utils.PytorchTestCase):
    device = torch.device('cpu')
