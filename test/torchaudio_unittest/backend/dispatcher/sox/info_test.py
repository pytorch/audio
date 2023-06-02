import itertools
import os
from functools import partial

from parameterized import parameterized
from torchaudio._backend.utils import get_info_func
from torchaudio._internal import module_utils as _mod_utils
from torchaudio_unittest.backend.common import get_encoding
from torchaudio_unittest.common_utils import (
    get_asset_path,
    get_wav_data,
    HttpServerMixin,
    PytorchTestCase,
    save_wav,
    skipIfNoExec,
    skipIfNoModule,
    skipIfNoSox,
    skipIfNoSoxDecoder,
    sox_utils,
    TempDirMixin,
)

from .common import name_func


if _mod_utils.is_module_available("requests"):
    import requests


@skipIfNoExec("sox")
@skipIfNoSox
class TestInfo(TempDirMixin, PytorchTestCase):
    _info = partial(get_info_func(), backend="sox")

    @parameterized.expand(
        list(
            itertools.product(
                ["float32", "int32", "int16", "uint8"],
                [8000, 16000],
                [1, 2],
            )
        ),
        name_func=name_func,
    )
    def test_wav(self, dtype, sample_rate, num_channels):
        """`self._info` can check wav file correctly"""
        duration = 1
        path = self.get_temp_path("data.wav")
        data = get_wav_data(dtype, num_channels, normalize=False, num_frames=duration * sample_rate)
        save_wav(path, data, sample_rate)
        info = self._info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels
        assert info.bits_per_sample == sox_utils.get_bit_depth(dtype)
        assert info.encoding == get_encoding("wav", dtype)

    @parameterized.expand(
        list(
            itertools.product(
                ["float32", "int32", "int16", "uint8"],
                [8000, 16000],
                [4, 8, 16, 32],
            )
        ),
        name_func=name_func,
    )
    def test_wav_multiple_channels(self, dtype, sample_rate, num_channels):
        """`self._info` can check wav file with channels more than 2 correctly"""
        duration = 1
        path = self.get_temp_path("data.wav")
        data = get_wav_data(dtype, num_channels, normalize=False, num_frames=duration * sample_rate)
        save_wav(path, data, sample_rate)
        info = self._info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels
        assert info.bits_per_sample == sox_utils.get_bit_depth(dtype)
        assert info.encoding == get_encoding("wav", dtype)

    @parameterized.expand(
        list(
            itertools.product(
                [8000, 16000],
                [1, 2],
                list(range(9)),
            )
        ),
        name_func=name_func,
    )
    def test_flac(self, sample_rate, num_channels, compression_level):
        """`self._info` can check flac file correctly"""
        duration = 1
        path = self.get_temp_path("data.flac")
        sox_utils.gen_audio_file(
            path,
            sample_rate,
            num_channels,
            compression=compression_level,
            duration=duration,
        )
        info = self._info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels
        assert info.bits_per_sample == 24  # FLAC standard
        assert info.encoding == "FLAC"

    @parameterized.expand(
        list(
            itertools.product(
                [8000, 16000],
                [1, 2],
                [-1, 0, 1, 2, 3, 3.6, 5, 10],
            )
        ),
        name_func=name_func,
    )
    def test_vorbis(self, sample_rate, num_channels, quality_level):
        """`self._info` can check vorbis file correctly"""
        duration = 1
        path = self.get_temp_path("data.vorbis")
        sox_utils.gen_audio_file(
            path,
            sample_rate,
            num_channels,
            compression=quality_level,
            duration=duration,
        )
        info = self._info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels
        assert info.bits_per_sample == 0  # bit_per_sample is irrelevant for compressed formats
        assert info.encoding == "VORBIS"

    @parameterized.expand(
        list(
            itertools.product(
                [8000, 16000],
                [1, 2],
                [16, 32],
            )
        ),
        name_func=name_func,
    )
    def test_sphere(self, sample_rate, num_channels, bits_per_sample):
        """`self._info` can check sph file correctly"""
        duration = 1
        path = self.get_temp_path("data.sph")
        sox_utils.gen_audio_file(path, sample_rate, num_channels, duration=duration, bit_depth=bits_per_sample)
        info = self._info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels
        assert info.bits_per_sample == bits_per_sample
        assert info.encoding == "PCM_S"

    @parameterized.expand(
        list(
            itertools.product(
                ["int32", "int16", "uint8"],
                [8000, 16000],
                [1, 2],
            )
        ),
        name_func=name_func,
    )
    def test_amb(self, dtype, sample_rate, num_channels):
        """`self._info` can check amb file correctly"""
        duration = 1
        path = self.get_temp_path("data.amb")
        bits_per_sample = sox_utils.get_bit_depth(dtype)
        sox_utils.gen_audio_file(path, sample_rate, num_channels, bit_depth=bits_per_sample, duration=duration)
        info = self._info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels
        assert info.bits_per_sample == bits_per_sample
        assert info.encoding == get_encoding("amb", dtype)

    @skipIfNoSoxDecoder("amr_nb")
    def test_amr_nb(self):
        """`self._info` can check amr-nb file correctly"""
        duration = 1
        num_channels = 1
        sample_rate = 8000
        path = self.get_temp_path("data.amr-nb")
        sox_utils.gen_audio_file(
            path, sample_rate=sample_rate, num_channels=num_channels, bit_depth=16, duration=duration
        )
        info = self._info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels
        assert info.bits_per_sample == 0
        assert info.encoding == "AMR_NB"

    def test_ulaw(self):
        """`self._info` can check ulaw file correctly"""
        duration = 1
        num_channels = 1
        sample_rate = 8000
        path = self.get_temp_path("data.wav")
        sox_utils.gen_audio_file(
            path, sample_rate=sample_rate, num_channels=num_channels, bit_depth=8, encoding="u-law", duration=duration
        )
        info = self._info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels
        assert info.bits_per_sample == 8
        assert info.encoding == "ULAW"

    def test_alaw(self):
        """`self._info` can check alaw file correctly"""
        duration = 1
        num_channels = 1
        sample_rate = 8000
        path = self.get_temp_path("data.wav")
        sox_utils.gen_audio_file(
            path, sample_rate=sample_rate, num_channels=num_channels, bit_depth=8, encoding="a-law", duration=duration
        )
        info = self._info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels
        assert info.bits_per_sample == 8
        assert info.encoding == "ALAW"

    def test_gsm(self):
        """`self._info` can check gsm file correctly"""
        duration = 1
        num_channels = 1
        sample_rate = 8000
        path = self.get_temp_path("data.gsm")
        sox_utils.gen_audio_file(path, sample_rate=sample_rate, num_channels=num_channels, duration=duration)
        info = self._info(path)
        assert info.sample_rate == sample_rate
        assert info.num_channels == num_channels
        assert info.bits_per_sample == 0
        assert info.encoding == "GSM"

    def test_htk(self):
        """`self._info` can check HTK file correctly"""
        duration = 1
        num_channels = 1
        sample_rate = 8000
        path = self.get_temp_path("data.htk")
        sox_utils.gen_audio_file(
            path, sample_rate=sample_rate, num_channels=num_channels, bit_depth=16, duration=duration
        )
        info = self._info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels
        assert info.bits_per_sample == 16
        assert info.encoding == "PCM_S"


@skipIfNoSoxDecoder("opus")
class TestInfoOpus(PytorchTestCase):
    _info = partial(get_info_func(), backend="sox")

    @parameterized.expand(
        list(
            itertools.product(
                ["96k"],
                [1, 2],
                [0, 5, 10],
            )
        ),
        name_func=name_func,
    )
    def test_opus(self, bitrate, num_channels, compression_level):
        """`self._info` can check opus file correcty"""
        path = get_asset_path("io", f"{bitrate}_{compression_level}_{num_channels}ch.opus")
        info = self._info(path)
        assert info.sample_rate == 48000
        assert info.num_frames == 32768
        assert info.num_channels == num_channels
        assert info.bits_per_sample == 0  # bit_per_sample is irrelevant for compressed formats
        assert info.encoding == "OPUS"


class FileObjTestBase(TempDirMixin):
    def _gen_file(self, ext, dtype, sample_rate, num_channels, num_frames, *, comments=None):
        path = self.get_temp_path(f"test.{ext}")
        bit_depth = sox_utils.get_bit_depth(dtype)
        duration = num_frames / sample_rate
        comment_file = self._gen_comment_file(comments) if comments else None

        sox_utils.gen_audio_file(
            path,
            sample_rate,
            num_channels=num_channels,
            encoding=sox_utils.get_encoding(dtype),
            bit_depth=bit_depth,
            duration=duration,
            comment_file=comment_file,
        )
        return path

    def _gen_comment_file(self, comments):
        comment_path = self.get_temp_path("comment.txt")
        with open(comment_path, "w") as file_:
            file_.writelines(comments)
        return comment_path


class Unseekable:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, n):
        return self.fileobj.read(n)


@skipIfNoSox
@skipIfNoExec("sox")
class TestFileObject(FileObjTestBase, PytorchTestCase):
    _info = partial(get_info_func(), backend="sox")

    def _query_fileobj(self, ext, dtype, sample_rate, num_channels, num_frames, *, comments=None):
        path = self._gen_file(ext, dtype, sample_rate, num_channels, num_frames, comments=comments)
        with open(path, "rb") as fileobj:
            return self._info(fileobj, None)

    @parameterized.expand(
        [
            ("wav", "float32"),
            ("wav", "int32"),
            ("wav", "int16"),
            ("wav", "uint8"),
            # ("mp3", "float32"),
            ("flac", "float32"),
            ("vorbis", "float32"),
            ("amb", "int16"),
        ]
    )
    def test_fileobj(self, ext, dtype):
        """Querying audio via file object works"""
        sample_rate = 16000
        num_frames = 3 * sample_rate
        num_channels = 2
        with self.assertRaisesRegex(ValueError, "SoX backend does not support reading"):
            self._query_fileobj(ext, dtype, sample_rate, num_channels, num_frames)


@skipIfNoSox
@skipIfNoExec("sox")
@skipIfNoModule("requests")
class TestFileObjectHttp(HttpServerMixin, FileObjTestBase, PytorchTestCase):
    _info = partial(get_info_func(), backend="sox")

    def _query_http(self, ext, dtype, sample_rate, num_channels, num_frames):
        audio_path = self._gen_file(ext, dtype, sample_rate, num_channels, num_frames)
        audio_file = os.path.basename(audio_path)

        url = self.get_url(audio_file)
        # format_ = ext if ext in ["mp3"] else None
        with requests.get(url, stream=True) as resp:
            return self._info(Unseekable(resp.raw), format=None)

    @parameterized.expand(
        [
            ("wav", "float32"),
            ("wav", "int32"),
            ("wav", "int16"),
            ("wav", "uint8"),
            # ("mp3", "float32"),
            ("flac", "float32"),
            ("vorbis", "float32"),
            ("amb", "int16"),
        ]
    )
    def test_requests(self, ext, dtype):
        """Querying compressed audio via requests works"""
        sample_rate = 16000
        num_frames = 3.0 * sample_rate
        num_channels = 2
        with self.assertRaisesRegex(ValueError, "SoX backend does not support reading"):
            self._query_http(ext, dtype, sample_rate, num_channels, num_frames)


@skipIfNoSox
class TestInfoNoSuchFile(PytorchTestCase):
    _info = partial(get_info_func(), backend="sox")

    def test_info_fail(self):
        """
        When attempted to get info on a non-existing file, error message must contain the file path.
        """
        path = "non_existing_audio.wav"
        with self.assertRaisesRegex(RuntimeError, path):
            self._info(path)
