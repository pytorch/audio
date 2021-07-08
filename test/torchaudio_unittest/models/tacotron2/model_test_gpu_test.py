import torch

from torchaudio_unittest.common_utils import skipIfNoCuda, PytorchTestCase
from .model_test_impl import Tacotron2Encoder, Tacotron2Decoder, Tacotron2


@skipIfNoCuda
class TestTacotron2EncoderFloat32(Tacotron2Encoder, PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cuda')


@skipIfNoCuda
class TestTacotron2DecoderFloat32(Tacotron2Decoder, PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cuda')


@skipIfNoCuda
class TestTacotron2Float32(Tacotron2, PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cuda')
