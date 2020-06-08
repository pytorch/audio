import unittest

import torchaudio
from torchaudio._internal.module_utils import is_module_available


class TestBackendSwitch(unittest.TestCase):
    def test_no_backend(self):
        torchaudio.set_audio_backend(None)
        assert torchaudio.load == torchaudio.backend.no_backend.load
        assert torchaudio.load_wav == torchaudio.backend.no_backend.load_wav
        assert torchaudio.save == torchaudio.backend.no_backend.save
        assert torchaudio.info == torchaudio.backend.no_backend.info
        assert torchaudio.get_audio_backend() is None

    @unittest.skipIf(
        not is_module_available('torchaudio._torchaudio'),
        'torchaudio C++ extension not available')
    def test_sox(self):
        torchaudio.set_audio_backend('sox')
        assert torchaudio.load == torchaudio.backend.sox_backend.load
        assert torchaudio.load_wav == torchaudio.backend.sox_backend.load_wav
        assert torchaudio.save == torchaudio.backend.sox_backend.save
        assert torchaudio.info == torchaudio.backend.sox_backend.info
        assert torchaudio.get_audio_backend() == 'sox'

    @unittest.skipIf(not is_module_available('soundfile'), '"soundfile" not available')
    def test_soundfile(self):
        torchaudio.set_audio_backend('soundfile')
        assert torchaudio.load == torchaudio.backend.soundfile_backend.load
        assert torchaudio.load_wav == torchaudio.backend.soundfile_backend.load_wav
        assert torchaudio.save == torchaudio.backend.soundfile_backend.save
        assert torchaudio.info == torchaudio.backend.soundfile_backend.info
        assert torchaudio.get_audio_backend() == 'soundfile'
