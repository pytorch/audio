import torch
from torchaudio_unittest.common_utils import PytorchTestCase

from .torchscript_consistency_test_impl import TorchScriptConsistencyCPUOnlyTestImpl, TorchScriptConsistencyTestImpl


class TorchScriptConsistencyCPUFloat32Test(TorchScriptConsistencyTestImpl, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")


class TorchScriptConsistencyCPUFloat64Test(TorchScriptConsistencyTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cpu")


class TorchScriptConsistencyCPUOnlyFloat32Test(TorchScriptConsistencyCPUOnlyTestImpl, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")


class TorchScriptConsistencyCPUOnlyFloat64Test(TorchScriptConsistencyCPUOnlyTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cpu")
