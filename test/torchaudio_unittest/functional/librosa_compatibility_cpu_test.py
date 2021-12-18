from torchaudio_unittest.common_utils import PytorchTestCase

from .librosa_compatibility_test_impl import Functional, FunctionalComplex


class TestFunctionalCPU(Functional, PytorchTestCase):
    device = "cpu"


class TestFunctionalComplexCPU(FunctionalComplex, PytorchTestCase):
    device = "cpu"
