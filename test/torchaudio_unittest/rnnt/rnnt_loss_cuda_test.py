import torch
from .rnnt_loss_impl import RNNTLossTest
from torchaudio_unittest import common_utils
from .utils import skipIfNoRNNT


@skipIfNoRNNT
@common_utils.skipIfNoCuda
class TestRNNTLoss(RNNTLossTest, common_utils.PytorchTestCase):
    device = torch.device('cuda')
