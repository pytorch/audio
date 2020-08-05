import torchaudio

from torchaudio_unittest import common_utils


class BackendSwitchMixin:
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


class TestBackendSwitch_NoBackend(BackendSwitchMixin, common_utils.TorchaudioTestCase):
    backend = None
    backend_module = torchaudio.backend.no_backend


@common_utils.skipIfNoExtension
class TestBackendSwitch_SoX(BackendSwitchMixin, common_utils.TorchaudioTestCase):
    backend = 'sox'
    backend_module = torchaudio.backend.sox_backend


@common_utils.skipIfNoExtension
class TestBackendSwitch_SoXIO(BackendSwitchMixin, common_utils.TorchaudioTestCase):
    backend = 'sox_io'
    backend_module = torchaudio.backend.sox_io_backend


@common_utils.skipIfNoModule('soundfile')
class TestBackendSwitch_soundfile(BackendSwitchMixin, common_utils.TorchaudioTestCase):
    backend = 'soundfile'
    backend_module = torchaudio.backend.soundfile_backend
