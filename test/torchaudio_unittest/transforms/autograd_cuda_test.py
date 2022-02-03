from torchaudio_unittest.common_utils import (
    PytorchTestCase,
    skipIfNoCuda,
)

from .autograd_test_impl import AutogradTestMixin, AutogradTestFloat32


@skipIfNoCuda
class AutogradCUDATest(AutogradTestMixin, PytorchTestCase):
    device = "cuda"


@skipIfNoCuda
class AutogradRNNTCUDATest(AutogradTestFloat32, PytorchTestCase):
    device = "cuda"
