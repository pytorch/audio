from torchaudio.utils import sox_utils

from torchaudio_unittest.common_utils import (
    PytorchTestCase,
    skipIfNoExtension,
)


@skipIfNoExtension
class TestSoxUtils(PytorchTestCase):
    """Smoke tests for sox_util module"""
    def test_set_seed(self):
        """`set_seed` does not crush"""
        sox_utils.set_seed(0)

    def test_set_verbosity(self):
        """`set_verbosity` does not crush"""
        for val in range(6, 0, -1):
            sox_utils.set_verbosity(val)

    def test_set_buffer_size(self):
        """`set_buffer_size` does not crush"""
        sox_utils.set_buffer_size(131072)
        # back to default
        sox_utils.set_buffer_size(8192)

    def test_set_use_threads(self):
        """`set_use_threads` does not crush"""
        sox_utils.set_use_threads(True)
        # back to default
        sox_utils.set_use_threads(False)

    def test_list_effects(self):
        """`list_effects` returns the list of available effects"""
        effects = sox_utils.list_effects()
        # We cannot infer what effects are available, so only check some of them.
        assert 'highpass' in effects
        assert 'phaser' in effects
        assert 'gain' in effects

    def test_list_read_formats(self):
        """`list_read_formats` returns the list of supported formats"""
        formats = sox_utils.list_read_formats()
        assert 'wav' in formats

    def test_list_write_formats(self):
        """`list_write_formats` returns the list of supported formats"""
        formats = sox_utils.list_write_formats()
        assert 'opus' not in formats
