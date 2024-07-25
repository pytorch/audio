from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoCuda

from .autograd_test_impl import AutogradTestFloat32, AutogradTestMixin


@skipIfNoCuda
class AutogradCUDATest(AutogradTestMixin, PytorchTestCase):
    device = "cuda"


@skipIfNoCuda
class AutogradRNNTCUDATest(AutogradTestFloat32, PytorchTestCase):
    device = "cuda"
