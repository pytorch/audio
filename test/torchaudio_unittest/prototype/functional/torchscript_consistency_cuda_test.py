import torch
from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoCuda

from .torchscript_consistency_test_impl import TorchScriptConsistencyTestImpl


@skipIfNoCuda
class TorchScriptConsistencyCUDAFloat32Test(TorchScriptConsistencyTestImpl, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cuda")


@skipIfNoCuda
class TorchScriptConsistencyCUDAFloat64Test(TorchScriptConsistencyTestImpl, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cuda")
