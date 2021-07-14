import os
import unittest

import torch

from .tacotron2_loss_impl import (
    Tacotron2LossShapeTests,
    Tacotron2LossTorchscriptTests,
    Tacotron2LossGradcheckTests,
)


def skipIfNoCuda(test_item):
    if torch.cuda.is_available():
        return test_item
    force_cuda_test = os.environ.get("TORCHAUDIO_TEST_FORCE_CUDA", "0")
    if force_cuda_test not in ["0", "1"]:
        raise ValueError('"TORCHAUDIO_TEST_FORCE_CUDA" must be either "0" or "1".')
    if force_cuda_test == "1":
        raise RuntimeError(
            '"TORCHAUDIO_TEST_FORCE_CUDA" is set but CUDA is not available.'
        )
    return unittest.skip("CUDA is not available.")(test_item)


@skipIfNoCuda
class TestTacotron2LossShapeFloat32CUDA(unittest.TestCase, Tacotron2LossShapeTests):
    dtype = torch.float32
    device = torch.device("cuda")


@skipIfNoCuda
class TestTacotron2TorchsciptFloat32CUDA(unittest.TestCase, Tacotron2LossTorchscriptTests):
    dtype = torch.float32
    device = torch.device("cuda")


@skipIfNoCuda
class TestTacotron2GradcheckFloat64CUDA(unittest.TestCase, Tacotron2LossGradcheckTests):
    dtype = torch.float64   # gradcheck needs a higher numerical accuracy
    device = torch.device("cuda")
