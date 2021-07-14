import torch
import unittest

from .tacotron2_loss_impl import (
    Tacotron2LossShapeTests,
    Tacotron2LossTorchscriptTests,
    Tacotron2LossGradcheckTests,
)


class TestTacotron2LossShapeFloat32CPU(unittest.TestCase, Tacotron2LossShapeTests):
    dtype = torch.float32
    device = torch.device("cpu")


class TestTacotron2TorchsciptFloat32CPU(unittest.TestCase, Tacotron2LossTorchscriptTests):
    dtype = torch.float32
    device = torch.device("cpu")


class TestTacotron2GradcheckFloat64CPU(unittest.TestCase, Tacotron2LossGradcheckTests):
    dtype = torch.float64   # gradcheck needs a higher numerical accuracy
    device = torch.device("cpu")
