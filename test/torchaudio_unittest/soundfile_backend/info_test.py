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

import pytest
from _pytest.monkeypatch import MonkeyPatch

if _mod_utils.is_module_available("soundfile"):
    import soundfile


@skipIfNoModule("soundfile")
class TestInfo(TempDirMixin, PytorchTestCase):
    @parameterize(
        [("float32", 32), ("int32", 32), ("int16", 16), ("uint8", 8)], [8000, 16000], [1, 2],
    )
    def test_wav(self, dtype_and_bit_depth, sample_rate, num_channels):
        """`soundfile_backend.info` can check wav file correctly"""
        dtype, bits_per_sample = dtype_and_bit_depth
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
        assert info.bits_per_sample == bits_per_sample

    @parameterize(
        [("float32", 32), ("int32", 32), ("int16", 16), ("uint8", 8)], [8000, 16000], [1, 2],
    )
    def test_wav_multiple_channels(self, dtype_and_bit_depth, sample_rate, num_channels):
        """`soundfile_backend.info` can check wav file with channels more than 2 correctly"""
        dtype, bits_per_sample = dtype_and_bit_depth
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
        assert info.bits_per_sample == bits_per_sample

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
        assert info.bits_per_sample == 16

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
        assert info.bits_per_sample == 0

    @parameterize([8000, 16000], [1, 2], [('PCM_24', 24), ('PCM_32', 32)])
    @skipIfFormatNotSupported("NIST")
    def test_sphere(self, sample_rate, num_channels, subtype_and_bit_depth):
        """`soundfile_backend.info` can check sph file correctly"""
        duration = 1
        num_frames = sample_rate * duration
        data = torch.randn(num_frames, num_channels).numpy()
        path = self.get_temp_path("data.nist")
        subtype, bits_per_sample = subtype_and_bit_depth
        soundfile.write(path, data, sample_rate, subtype=subtype)

        info = soundfile_backend.info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels
        assert info.bits_per_sample == bits_per_sample


def test_unknown_subtype_warning(tmp_path, monkeypatch):
    """soundfile_backend.info issues a warning when the subtype is unknown

    This will happen if a new subtype is supported in SoundFile: the _SUBTYPE_TO_BITS_PER_SAMPLE
    dict should be updated.
    """

    soundfile_info_original = soundfile.info

    def info_wrapper(filepath):
        # Wraps soundfile.info and sets the subtype to some unknown value
        sinfo = soundfile_info_original(filepath)
        sinfo.subtype = 'SOME_UNKNOWN_SUBTYPE'
        return sinfo

    monkeypatch.setattr(soundfile, "info", info_wrapper)

    data = get_wav_data(
        dtype='float32', num_channels=1, normalize=False, num_frames=16000
    )
    path = tmp_path / 'data.wav'
    save_wav(path, data, sample_rate=16000)
    with pytest.warns(UserWarning, match="subtype is unknown to TorchAudio"):
        info = soundfile_backend.info(path)
    assert info.bits_per_sample == 0
