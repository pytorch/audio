from torchaudio_unittest.common_utils import PytorchTestCase

from .autograd_test_impl import AutogradTestFloat32, AutogradTestMixin


class AutogradCPUTest(AutogradTestMixin, PytorchTestCase):
    device = "cpu"


class AutogradRNNTCPUTest(AutogradTestFloat32, PytorchTestCase):
    device = "cpu"
