from torchaudio_unittest.common_utils import (
    PytorchTestCase,
    skipIfNoCuda,
)
from .autograd_test_impl import AutogradTestMixin


@skipIfNoCuda
class AutogradCUDATest(AutogradTestMixin, PytorchTestCase):
    device = 'cuda'
