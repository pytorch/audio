from torchaudio_unittest.common_utils import PytorchTestCase
from .autograd_test_impl import AutogradTestMixin


class AutogradCPUTest(AutogradTestMixin, PytorchTestCase):
    device = 'cpu'
