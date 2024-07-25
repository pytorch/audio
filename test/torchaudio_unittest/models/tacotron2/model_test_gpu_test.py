import torch
from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoCuda

from .model_test_impl import Tacotron2DecoderTests, Tacotron2EncoderTests, Tacotron2Tests


@skipIfNoCuda
class TestTacotron2EncoderFloat32CUDA(Tacotron2EncoderTests, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cuda")


@skipIfNoCuda
class TestTacotron2DecoderFloat32CUDA(Tacotron2DecoderTests, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cuda")


@skipIfNoCuda
class TestTacotron2Float32CUDA(Tacotron2Tests, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cuda")
