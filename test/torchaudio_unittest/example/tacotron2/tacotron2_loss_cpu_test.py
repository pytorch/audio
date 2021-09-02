import torch

from .tacotron2_loss_impl import (
    Tacotron2LossShapeTests,
    Tacotron2LossTorchscriptTests,
    Tacotron2LossGradcheckTests,
)
from torchaudio_unittest.common_utils import PytorchTestCase


class TestTacotron2LossShapeFloat32CPU(Tacotron2LossShapeTests, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")


class TestTacotron2TorchsciptFloat32CPU(Tacotron2LossTorchscriptTests, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")


class TestTacotron2GradcheckFloat64CPU(Tacotron2LossGradcheckTests, PytorchTestCase):
    dtype = torch.float64   # gradcheck needs a higher numerical accuracy
    device = torch.device("cpu")
