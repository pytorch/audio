from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoCuda

from .autograd_test_impl import Autograd


@skipIfNoCuda
class AutogradCUDATest(Autograd, PytorchTestCase):
    device = "cuda"
