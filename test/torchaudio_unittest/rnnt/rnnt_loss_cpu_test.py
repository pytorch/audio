import torch
from torchaudio_unittest import common_utils
from .utils import skipIfNoTransducer
from .rnnt_loss_impl import RNNTLossTest


@skipIfNoTransducer
class TestRNNTLoss(RNNTLossTest, common_utils.PytorchTestCase):
    device = torch.device('cpu')
