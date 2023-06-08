from torchaudio_unittest.common_utils import PytorchTestCase

from .librosa_compatibility_test_impl import Functional


class TestFunctionalCPU(Functional, PytorchTestCase):
    device = "cpu"
