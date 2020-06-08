import unittest

import torchaudio
from torchaudio._internal.module_utils import is_module_available


class BackendSwitch:
    """Test set/get_audio_backend works"""
    backend = None
    backend_module = None

    def test_switch(self):
        torchaudio.set_audio_backend(self.backend)
        if self.backend is None:
            assert torchaudio.get_audio_backend() is None
        else:
            assert torchaudio.get_audio_backend() == self.backend
        assert torchaudio.load == self.backend_module.load
        assert torchaudio.load_wav == self.backend_module.load_wav
        assert torchaudio.save == self.backend_module.save
        assert torchaudio.info == self.backend_module.info


class TestBackendSwitch_NoBackend(BackendSwitch, unittest.TestCase):
    backend = None
    backend_module = torchaudio.backend.no_backend


@unittest.skipIf(
    not is_module_available('torchaudio._torchaudio'),
    'torchaudio C++ extension not available')
class TestBackendSwitch_SoX(BackendSwitch, unittest.TestCase):
    backend = 'sox'
    backend_module = torchaudio.backend.sox_backend


@unittest.skipIf(not is_module_available('soundfile'), '"soundfile" not available')
class TestBackendSwitch_soundfile(BackendSwitch, unittest.TestCase):
    backend = 'soundfile'
    backend_module = torchaudio.backend.soundfile_backend
