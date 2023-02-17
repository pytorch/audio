import io
from unittest.mock import patch

import torch

from parameterized import parameterized
from torchaudio._backend.utils import (
    FFmpegBackend,
    get_info_func,
    get_load_func,
    get_save_func,
    SoundfileBackend,
    SoXBackend,
)
from torchaudio_unittest.common_utils import PytorchTestCase


class DispatcherTest(PytorchTestCase):
    @parameterized.expand(
        [
            # FFmpeg backend is used when no backend is specified.
            ({"ffmpeg": FFmpegBackend, "sox": SoXBackend, "soundfile": SoundfileBackend}, FFmpegBackend),
            # SoX backend is used when no backend is specified and FFmpeg is not available.
            ({"sox": SoXBackend, "soundfile": SoundfileBackend}, SoXBackend),
        ]
    )
    def test_info(self, available_backends, expected_backend):
        filename = "test.wav"
        format = "wav"
        with patch("torchaudio._backend.utils.get_available_backends", return_value=available_backends), patch(
            f"torchaudio._backend.utils.{expected_backend.__name__}.info"
        ) as mock_info:
            get_info_func()(filename, format=format)
            mock_info.assert_called_once_with(filename, format, 4096)

    @parameterized.expand(
        [
            # FFmpeg backend is used when no backend is specified.
            ({"ffmpeg": FFmpegBackend, "sox": SoXBackend, "soundfile": SoundfileBackend}, FFmpegBackend),
            # Soundfile backend is used when no backend is specified, FFmpeg is not available,
            # and input is file-like object (i.e. SoX is properly skipped over).
            ({"sox": SoXBackend, "soundfile": SoundfileBackend}, SoundfileBackend),
        ]
    )
    def test_info_fileobj(self, available_backends, expected_backend):
        f = io.BytesIO()
        format = "wav"
        buffer_size = 8192
        with patch("torchaudio._backend.utils.get_available_backends", return_value=available_backends), patch(
            f"torchaudio._backend.utils.{expected_backend.__name__}.info"
        ) as mock_info:
            get_info_func()(f, format=format, buffer_size=buffer_size)
            mock_info.assert_called_once_with(f, format, buffer_size)

    @parameterized.expand(
        [
            # FFmpeg backend is used when no backend is specified.
            ({"ffmpeg": FFmpegBackend, "sox": SoXBackend, "soundfile": SoundfileBackend}, FFmpegBackend),
            # SoX backend is used when no backend is specified and FFmpeg is not available.
            ({"sox": SoXBackend, "soundfile": SoundfileBackend}, SoXBackend),
        ]
    )
    def test_load(self, available_backends, expected_backend):
        filename = "test.wav"
        format = "wav"
        with patch("torchaudio._backend.utils.get_available_backends", return_value=available_backends), patch(
            f"torchaudio._backend.utils.{expected_backend.__name__}.load"
        ) as mock_load:
            get_load_func()(filename, format=format)
            mock_load.assert_called_once_with(filename, 0, -1, True, True, format, 4096)

    @parameterized.expand(
        [
            # FFmpeg backend is used when no backend is specified.
            ({"ffmpeg": FFmpegBackend, "sox": SoXBackend, "soundfile": SoundfileBackend}, FFmpegBackend),
            # Soundfile backend is used when no backend is specified, FFmpeg is not available,
            # and input is file-like object (i.e. SoX is properly skipped over).
            ({"sox": SoXBackend, "soundfile": SoundfileBackend}, SoundfileBackend),
        ]
    )
    def test_load_fileobj(self, available_backends, expected_backend):
        f = io.BytesIO()
        format = "wav"
        buffer_size = 8192
        with patch("torchaudio._backend.utils.get_available_backends", return_value=available_backends), patch(
            f"torchaudio._backend.utils.{expected_backend.__name__}.load"
        ) as mock_load:
            get_load_func()(f, format=format, buffer_size=buffer_size)
            mock_load.assert_called_once_with(f, 0, -1, True, True, format, buffer_size)

    @parameterized.expand(
        [
            # FFmpeg backend is used when no backend is specified.
            ({"ffmpeg": FFmpegBackend, "sox": SoXBackend, "soundfile": SoundfileBackend}, FFmpegBackend),
            # SoX backend is used when no backend is specified and FFmpeg is not available.
            ({"sox": SoXBackend, "soundfile": SoundfileBackend}, SoXBackend),
        ]
    )
    def test_save(self, available_backends, expected_backend):
        src = torch.zeros((2, 10))
        filename = "test.wav"
        format = "wav"
        sample_rate = 16000
        with patch("torchaudio._backend.utils.get_available_backends", return_value=available_backends), patch(
            f"torchaudio._backend.utils.{expected_backend.__name__}.save"
        ) as mock_save:
            get_save_func()(filename, src, sample_rate, format=format)
            mock_save.assert_called_once_with(filename, src, sample_rate, True, format, None, None, 4096)

    @parameterized.expand(
        [
            # FFmpeg backend is used when no backend is specified.
            ({"ffmpeg": FFmpegBackend, "sox": SoXBackend, "soundfile": SoundfileBackend}, FFmpegBackend),
            # Soundfile backend is used when no backend is specified, FFmpeg is not available,
            # and input is file-like object (i.e. SoX is properly skipped over).
            ({"sox": SoXBackend, "soundfile": SoundfileBackend}, SoundfileBackend),
        ]
    )
    def test_save_fileobj(self, available_backends, expected_backend):
        src = torch.zeros((2, 10))
        f = io.BytesIO()
        format = "wav"
        buffer_size = 8192
        sample_rate = 16000
        with patch("torchaudio._backend.utils.get_available_backends", return_value=available_backends), patch(
            f"torchaudio._backend.utils.{expected_backend.__name__}.save"
        ) as mock_save:
            get_save_func()(f, src, sample_rate, format=format, buffer_size=buffer_size)
            mock_save.assert_called_once_with(f, src, sample_rate, True, format, None, None, buffer_size)
