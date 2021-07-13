import torch

from torchaudio_unittest.common_utils import PytorchTestCase
from .model_test_impl import (
    Tacotron2EncoderTests,
    Tacotron2DecoderTests,
    Tacotron2Tests,
)


class TestTacotron2EncoderFloat32(Tacotron2EncoderTests, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")


class TestTacotron2DecoderFloat32(Tacotron2DecoderTests, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")


class TestTacotron2Float32(Tacotron2Tests, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")
