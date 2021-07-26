import torch

from .tacotron2_loss_impl import (
    Tacotron2LossShapeTests,
    Tacotron2LossTorchscriptTests,
    Tacotron2LossGradcheckTests,
)
from torchaudio_unittest.common_utils import PytorchTestCase


class TestTacotron2LossShapeFloat32CPU(PytorchTestCase, Tacotron2LossShapeTests):
    dtype = torch.float32
    device = torch.device("cpu")


class TestTacotron2TorchsciptFloat32CPU(PytorchTestCase, Tacotron2LossTorchscriptTests):
    dtype = torch.float32
    device = torch.device("cpu")


class TestTacotron2GradcheckFloat64CPU(PytorchTestCase, Tacotron2LossGradcheckTests):
    dtype = torch.float64   # gradcheck needs a higher numerical accuracy
    device = torch.device("cpu")
