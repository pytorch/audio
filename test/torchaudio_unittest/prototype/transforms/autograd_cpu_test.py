from torchaudio_unittest.common_utils import PytorchTestCase

from .autograd_test_impl import Autograd


class AutogradCPUTest(Autograd, PytorchTestCase):
    device = "cpu"
