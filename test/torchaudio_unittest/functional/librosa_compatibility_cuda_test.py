from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoCuda
from .librosa_compatibility_test_impl import Functional, FunctionalComplex


@skipIfNoCuda
class TestFunctionalCUDA(Functional, PytorchTestCase):
    device = 'cuda'


@skipIfNoCuda
class TestFunctionalComplexCPU(FunctionalComplex, PytorchTestCase):
    device = 'cpu'
