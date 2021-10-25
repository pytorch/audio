import torch

from .tacotron2_loss_impl import (
    Tacotron2LossShapeTests,
    Tacotron2LossTorchscriptTests,
    Tacotron2LossGradcheckTests,
)
from torchaudio_unittest.common_utils import skipIfNoCuda, PytorchTestCase


@skipIfNoCuda
class TestTacotron2LossShapeFloat32CUDA(PytorchTestCase, Tacotron2LossShapeTests):
    dtype = torch.float32
    device = torch.device("cuda")


@skipIfNoCuda
class TestTacotron2TorchsciptFloat32CUDA(PytorchTestCase, Tacotron2LossTorchscriptTests):
    dtype = torch.float32
    device = torch.device("cuda")


@skipIfNoCuda
class TestTacotron2GradcheckFloat64CUDA(PytorchTestCase, Tacotron2LossGradcheckTests):
    dtype = torch.float64   # gradcheck needs a higher numerical accuracy
    device = torch.device("cuda")
