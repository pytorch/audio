import torch
from torchaudio_unittest.common_utils import PytorchTestCase

from .torchscript_consistency_test_impl import TorchScriptConsistencyTestImpl


class TorchScriptConsistencyCPUFloat32Test(TorchScriptConsistencyTestImpl, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")


class TorchScriptConsistencyCPUFloat64Test(TorchScriptConsistencyTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cpu")
