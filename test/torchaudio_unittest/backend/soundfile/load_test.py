import os
import tarfile
from unittest.mock import patch

import torch
from torchaudio._internal import module_utils as _mod_utils
from torchaudio.backend import soundfile_backend
from parameterized import parameterized

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    PytorchTestCase,
    skipIfNoModule,
    get_wav_data,
    normalize_wav,
    load_wav,
    save_wav,
)
from .common import (
    parameterize,
    dtype2subtype,
    skipIfFormatNotSupported,
)

if _mod_utils.is_module_available("soundfile"):
    import soundfile


def _get_mock_path(
    ext: str, dtype: str, sample_rate: int, num_channels: int, num_frames: int,
):
    return f"{dtype}_{sample_rate}_{num_channels}_{num_frames}.{ext}"


def _get_mock_params(path: str):
    filename, ext = path.split(".")
    parts = filename.split("_")
    return {
        "ext": ext,
        "dtype": parts[0],
        "sample_rate": int(parts[1]),
        "num_channels": int(parts[2]),
        "num_frames": int(parts[3]),
    }


class SoundFileMock:
    def __init__(self, path, mode):
        assert mode == "r"
        self.path = path
        self._params = _get_mock_params(path)
        self._start = None

    @property
    def samplerate(self):
        return self._params["sample_rate"]

    @property
    def format(self):
        if self._params["ext"] == "wav":
            return "WAV"
        if self._params["ext"] == "flac":
            return "FLAC"
        if self._params["ext"] == "ogg":
            return "OGG"
        if self._params["ext"] in ["sph", "nis", "nist"]:
            return "NIST"

    @property
    def subtype(self):
        if self._params["ext"] == "ogg":
            return "VORBIS"
        return dtype2subtype(self._params["dtype"])

    def _prepare_read(self, start, stop, frames):
        assert stop is None
        self._start = start
        return frames

    def read(self, frames, dtype, always_2d):
        assert always_2d
        data = get_wav_data(
            dtype,
            self._params["num_channels"],
            normalize=False,
            num_frames=self._params["num_frames"],
            channels_first=False,
        ).numpy()
        return data[self._start:self._start + frames]

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass


class MockedLoadTest(PytorchTestCase):
    def assert_dtype(
        self, ext, dtype, sample_rate, num_channels, normalize, channels_first
    ):
        """When format is WAV or NIST, normalize=False will return the native dtype Tensor, otherwise float32"""
        num_frames = 3 * sample_rate
        path = _get_mock_path(ext, dtype, sample_rate, num_channels, num_frames)
        expected_dtype = (
            torch.float32
            if normalize or ext not in ["wav", "nist"]
            else getattr(torch, dtype)
        )
        with patch("soundfile.SoundFile", SoundFileMock):
            found, sr = soundfile_backend.load(
                path, normalize=normalize, channels_first=channels_first
            )
            assert found.dtype == expected_dtype
            assert sample_rate == sr

    @parameterize(
        ["uint8", "int16", "int32", "float32", "float64"],
        [8000, 16000],
        [1, 2],
        [True, False],
        [True, False],
    )
    def test_wav(self, dtype, sample_rate, num_channels, normalize, channels_first):
        """Returns native dtype when normalize=False else float32"""
        self.assert_dtype(
            "wav", dtype, sample_rate, num_channels, normalize, channels_first
        )

    @parameterize(
        ["int8", "int16", "int32"], [8000, 16000], [1, 2], [True, False], [True, False],
    )
    def test_sphere(self, dtype, sample_rate, num_channels, normalize, channels_first):
        """Returns float32 always"""
        self.assert_dtype(
            "sph", dtype, sample_rate, num_channels, normalize, channels_first
        )

    @parameterize([8000, 16000], [1, 2], [True, False], [True, False])
    def test_ogg(self, sample_rate, num_channels, normalize, channels_first):
        """Returns float32 always"""
        self.assert_dtype(
            "ogg", "int16", sample_rate, num_channels, normalize, channels_first
        )

    @parameterize([8000, 16000], [1, 2], [True, False], [True, False])
    def test_flac(self, sample_rate, num_channels, normalize, channels_first):
        """`soundfile_backend.load` can load ogg format."""
        self.assert_dtype(
            "flac", "int16", sample_rate, num_channels, normalize, channels_first
        )


class LoadTestBase(TempDirMixin, PytorchTestCase):
    def assert_wav(
        self,
        dtype,
        sample_rate,
        num_channels,
        normalize,
        channels_first=True,
        duration=1,
    ):
        """`soundfile_backend.load` can load wav format correctly.

        Wav data loaded with soundfile backend should match those with scipy
        """
        path = self.get_temp_path("reference.wav")
        num_frames = duration * sample_rate
        data = get_wav_data(
            dtype,
            num_channels,
            normalize=normalize,
            num_frames=num_frames,
            channels_first=channels_first,
        )
        save_wav(path, data, sample_rate, channels_first=channels_first)
        expected = load_wav(path, normalize=normalize, channels_first=channels_first)[0]
        data, sr = soundfile_backend.load(
            path, normalize=normalize, channels_first=channels_first
        )
        assert sr == sample_rate
        self.assertEqual(data, expected)

    def assert_sphere(
        self, dtype, sample_rate, num_channels, channels_first=True, duration=1,
    ):
        """`soundfile_backend.load` can load SPHERE format correctly."""
        path = self.get_temp_path("reference.sph")
        num_frames = duration * sample_rate
        raw = get_wav_data(
            dtype,
            num_channels,
            num_frames=num_frames,
            normalize=False,
            channels_first=False,
        )
        soundfile.write(
            path, raw, sample_rate, subtype=dtype2subtype(dtype), format="NIST"
        )
        expected = normalize_wav(raw.t() if channels_first else raw)
        data, sr = soundfile_backend.load(path, channels_first=channels_first)
        assert sr == sample_rate
        self.assertEqual(data, expected, atol=1e-4, rtol=1e-8)

    def assert_flac(
        self, dtype, sample_rate, num_channels, channels_first=True, duration=1,
    ):
        """`soundfile_backend.load` can load FLAC format correctly."""
        path = self.get_temp_path("reference.flac")
        num_frames = duration * sample_rate
        raw = get_wav_data(
            dtype,
            num_channels,
            num_frames=num_frames,
            normalize=False,
            channels_first=False,
        )
        soundfile.write(path, raw, sample_rate)
        expected = normalize_wav(raw.t() if channels_first else raw)
        data, sr = soundfile_backend.load(path, channels_first=channels_first)
        assert sr == sample_rate
        self.assertEqual(data, expected, atol=1e-4, rtol=1e-8)


@skipIfNoModule("soundfile")
class TestLoad(LoadTestBase):
    """Test the correctness of `soundfile_backend.load` for various formats"""

    @parameterize(
        ["float32", "int32", "int16"],
        [8000, 16000],
        [1, 2],
        [False, True],
        [False, True],
    )
    def test_wav(self, dtype, sample_rate, num_channels, normalize, channels_first):
        """`soundfile_backend.load` can load wav format correctly."""
        self.assert_wav(dtype, sample_rate, num_channels, normalize, channels_first)

    @parameterize(
        ["int16"], [16000], [2], [False],
    )
    def test_wav_large(self, dtype, sample_rate, num_channels, normalize):
        """`soundfile_backend.load` can load large wav file correctly."""
        two_hours = 2 * 60 * 60
        self.assert_wav(dtype, sample_rate, num_channels, normalize, duration=two_hours)

    @parameterize(["float32", "int32", "int16"], [4, 8, 16, 32], [False, True])
    def test_multiple_channels(self, dtype, num_channels, channels_first):
        """`soundfile_backend.load` can load wav file with more than 2 channels."""
        sample_rate = 8000
        normalize = False
        self.assert_wav(dtype, sample_rate, num_channels, normalize, channels_first)

    @parameterize(["int32", "int16"], [8000, 16000], [1, 2], [False, True])
    @skipIfFormatNotSupported("NIST")
    def test_sphere(self, dtype, sample_rate, num_channels, channels_first):
        """`soundfile_backend.load` can load sphere format correctly."""
        self.assert_sphere(dtype, sample_rate, num_channels, channels_first)

    @parameterize(["int32", "int16"], [8000, 16000], [1, 2], [False, True])
    @skipIfFormatNotSupported("FLAC")
    def test_flac(self, dtype, sample_rate, num_channels, channels_first):
        """`soundfile_backend.load` can load flac format correctly."""
        self.assert_flac(dtype, sample_rate, num_channels, channels_first)


@skipIfNoModule("soundfile")
class TestLoadFormat(TempDirMixin, PytorchTestCase):
    """Given `format` parameter, `so.load` can load files without extension"""
    original = None
    path = None

    def _make_file(self, format_):
        sample_rate = 8000
        path_with_ext = self.get_temp_path(f'test.{format_}')
        data = get_wav_data('float32', num_channels=2).numpy().T
        soundfile.write(path_with_ext, data, sample_rate)
        expected = soundfile.read(path_with_ext, dtype='float32')[0].T
        path = os.path.splitext(path_with_ext)[0]
        os.rename(path_with_ext, path)
        return path, expected

    def _test_format(self, format_):
        """Providing format allows to read file without extension"""
        path, expected = self._make_file(format_)
        found, _ = soundfile_backend.load(path)
        self.assertEqual(found, expected)

    @parameterized.expand([
        ('WAV', ), ('wav', ),
    ])
    def test_wav(self, format_):
        self._test_format(format_)

    @parameterized.expand([
        ('FLAC', ), ('flac',),
    ])
    @skipIfFormatNotSupported("FLAC")
    def test_flac(self, format_):
        self._test_format(format_)


@skipIfNoModule("soundfile")
class TestFileObject(TempDirMixin, PytorchTestCase):
    def _test_fileobj(self, ext):
        """Loading audio via file-like object works"""
        sample_rate = 16000
        path = self.get_temp_path(f'test.{ext}')

        data = get_wav_data('float32', num_channels=2).numpy().T
        soundfile.write(path, data, sample_rate)
        expected = soundfile.read(path, dtype='float32')[0].T

        with open(path, 'rb') as fileobj:
            found, sr = soundfile_backend.load(fileobj)
        assert sr == sample_rate
        self.assertEqual(expected, found)

    def test_fileobj_wav(self):
        """Loading audio via file-like object works"""
        self._test_fileobj('wav')

    @skipIfFormatNotSupported("FLAC")
    def test_fileobj_flac(self):
        """Loading audio via file-like object works"""
        self._test_fileobj('flac')

    def _test_tarfile(self, ext):
        """Loading audio via file-like object works"""
        sample_rate = 16000
        audio_file = f'test.{ext}'
        audio_path = self.get_temp_path(audio_file)
        archive_path = self.get_temp_path('archive.tar.gz')

        data = get_wav_data('float32', num_channels=2).numpy().T
        soundfile.write(audio_path, data, sample_rate)
        expected = soundfile.read(audio_path, dtype='float32')[0].T

        with tarfile.TarFile(archive_path, 'w') as tarobj:
            tarobj.add(audio_path, arcname=audio_file)
        with tarfile.TarFile(archive_path, 'r') as tarobj:
            fileobj = tarobj.extractfile(audio_file)
            found, sr = soundfile_backend.load(fileobj)

        assert sr == sample_rate
        self.assertEqual(expected, found)

    def test_tarfile_wav(self):
        """Loading audio via file-like object works"""
        self._test_tarfile('wav')

    @skipIfFormatNotSupported("FLAC")
    def test_tarfile_flac(self):
        """Loading audio via file-like object works"""
        self._test_tarfile('flac')
