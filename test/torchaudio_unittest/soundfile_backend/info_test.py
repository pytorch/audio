import torch
from torchaudio.backend import _soundfile_backend as soundfile_backend
from torchaudio._internal import module_utils as _mod_utils

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    PytorchTestCase,
    skipIfNoModule,
    get_wav_data,
    save_wav,
)
from .common import skipIfFormatNotSupported, parameterize

if _mod_utils.is_module_available("soundfile"):
    import soundfile


@skipIfNoModule("soundfile")
class TestInfo(TempDirMixin, PytorchTestCase):
    @parameterize(
        ["float32", "int32", "int16", "uint8"], [8000, 16000], [1, 2],
    )
    def test_wav(self, dtype, sample_rate, num_channels):
        """`soundfile_backend.info` can check wav file correctly"""
        duration = 1
        path = self.get_temp_path("data.wav")
        data = get_wav_data(
            dtype, num_channels, normalize=False, num_frames=duration * sample_rate
        )
        save_wav(path, data, sample_rate)
        info = soundfile_backend.info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels

    @parameterize(
        ["float32", "int32", "int16", "uint8"], [8000, 16000], [4, 8, 16, 32],
    )
    def test_wav_multiple_channels(self, dtype, sample_rate, num_channels):
        """`soundfile_backend.info` can check wav file with channels more than 2 correctly"""
        duration = 1
        path = self.get_temp_path("data.wav")
        data = get_wav_data(
            dtype, num_channels, normalize=False, num_frames=duration * sample_rate
        )
        save_wav(path, data, sample_rate)
        info = soundfile_backend.info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels

    @parameterize([8000, 16000], [1, 2])
    @skipIfFormatNotSupported("FLAC")
    def test_flac(self, sample_rate, num_channels):
        """`soundfile_backend.info` can check flac file correctly"""
        duration = 1
        num_frames = sample_rate * duration
        data = torch.randn(num_frames, num_channels).numpy()
        path = self.get_temp_path("data.flac")
        soundfile.write(path, data, sample_rate)

        info = soundfile_backend.info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == num_frames
        assert info.num_channels == num_channels

    @parameterize([8000, 16000], [1, 2])
    @skipIfFormatNotSupported("OGG")
    def test_ogg(self, sample_rate, num_channels):
        """`soundfile_backend.info` can check ogg file correctly"""
        duration = 1
        num_frames = sample_rate * duration
        data = torch.randn(num_frames, num_channels).numpy()
        path = self.get_temp_path("data.ogg")
        soundfile.write(path, data, sample_rate)

        info = soundfile_backend.info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels

    @parameterize([8000, 16000], [1, 2])
    @skipIfFormatNotSupported("NIST")
    def test_sphere(self, sample_rate, num_channels):
        """`soundfile_backend.info` can check sph file correctly"""
        duration = 1
        num_frames = sample_rate * duration
        data = torch.randn(num_frames, num_channels).numpy()
        path = self.get_temp_path("data.nist")
        soundfile.write(path, data, sample_rate)

        info = soundfile_backend.info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels
