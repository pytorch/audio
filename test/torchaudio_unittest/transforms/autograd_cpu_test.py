from torchaudio_unittest.common_utils import PytorchTestCase

from .autograd_test_impl import AutogradTestMixin, AutogradTestFloat32


class AutogradCPUTest(AutogradTestMixin, PytorchTestCase):
    device = "cpu"


class AutogradRNNTCPUTest(AutogradTestFloat32, PytorchTestCase):
    device = "cpu"
