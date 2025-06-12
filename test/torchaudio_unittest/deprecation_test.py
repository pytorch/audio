from torchaudio_unittest.common_utils import PytorchTestCase
from torchaudio._internal.module_utils import UNSUPPORTED


class TestDeprecations(PytorchTestCase):
    def test_deprecations(self):
        for func in UNSUPPORTED:
            with self.assertWarnsRegex(UserWarning, r"deprecated"):
                try:
                    func()
                except:
                    pass
