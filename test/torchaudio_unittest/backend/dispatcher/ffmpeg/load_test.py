import io
import itertools
import pathlib
import tarfile
from functools import partial

from parameterized import parameterized
from torchaudio._backend.ffmpeg import _parse_save_args
from torchaudio._backend.utils import get_load_func
from torchaudio._internal import module_utils as _mod_utils

from torchaudio_unittest.backend.dispatcher.sox.common import name_func
from torchaudio_unittest.common_utils import (
    disabledInCI,
    get_asset_path,
    get_wav_data,
    HttpServerMixin,
    load_wav,
    PytorchTestCase,
    save_wav,
    skipIfNoExec,
    skipIfNoFFmpeg,
    skipIfNoModule,
    sox_utils,
    TempDirMixin,
)

from .save_test import _convert_audio_file


if _mod_utils.is_module_available("requests"):
    import requests


class LoadTestBase(TempDirMixin, PytorchTestCase):
    _load = partial(get_load_func(), backend="ffmpeg")

    def assert_format(
        self,
        format: str,
        sample_rate: float,
        num_channels: int,
        compression: float = None,
        bit_depth: int = None,
        duration: float = 1,
        normalize: bool = True,
        encoding: str = None,
        atol: float = 4e-05,
        rtol: float = 1.3e-06,
    ):
        """`self._load` can load given format correctly.

        file encodings introduce delay and boundary effects so
        we create a reference wav file from the original file format

         x
         |
         |    1. Generate given format with Sox
         |
         + ----------------------------------+ 3. Convert to wav with FFmpeg
         |                                   |
         |    2. Load the given format       | 4. Load with scipy
         |       with torchaudio             |
         v                                   v
        tensor ----------> x <----------- tensor
                       5. Compare

        Underlying assumptions are;
        i. Conversion of given format to wav with FFmpeg preserves data.
        ii. Loading wav file with scipy is correct.

        By combining i & ii, step 2. and 4. allow for loading reference given format
        data without using torchaudio
        """
        path = self.get_temp_path(f"1.original.{format}")
        ref_path = self.get_temp_path("2.reference.wav")

        # 1. Generate the given format with sox
        sox_utils.gen_audio_file(
            path,
            sample_rate,
            num_channels,
            encoding=encoding,
            compression=compression,
            bit_depth=bit_depth,
            duration=duration,
        )
        # 2. Load the given format with torchaudio
        data, sr = self._load(path, normalize=normalize)

        # 3. Convert to wav with ffmpeg
        if normalize:
            encoder = "pcm_f32le"
        else:
            encoding_map = {
                "floating-point": "PCM_F",
                "signed-integer": "PCM_S",
                "unsigned-integer": "PCM_U",
            }
            _, encoder, _ = _parse_save_args(format, format, encoding_map.get(encoding), bit_depth)
        _convert_audio_file(path, ref_path, encoder=encoder)

        # 4. Load wav with scipy
        data_ref = load_wav(ref_path, normalize=normalize)[0]
        # 5. Compare
        assert sr == sample_rate
        self.assertEqual(data, data_ref, atol=atol, rtol=rtol)

    def assert_wav(self, dtype, sample_rate, num_channels, normalize, duration):
        """`self._load` can load wav format correctly.

        Wav data loaded with sox_io backend should match those with scipy
        """
        path = self.get_temp_path("reference.wav")
        data = get_wav_data(dtype, num_channels, normalize=normalize, num_frames=duration * sample_rate)
        save_wav(path, data, sample_rate)
        expected = load_wav(path, normalize=normalize)[0]
        data, sr = self._load(path, normalize=normalize)
        assert sr == sample_rate
        self.assertEqual(data, expected)


@skipIfNoExec("sox")
@skipIfNoFFmpeg
class TestLoad(LoadTestBase):
    """Test the correctness of `self._load` for various formats"""

    def test_pathlike(self):
        """FFmpeg dispatcher can load waveform from pathlike object"""
        sample_rate = 16000
        dtype = "float32"
        num_channels = 2
        duration = 1

        path = self.get_temp_path("data.wav")
        data = get_wav_data(dtype, num_channels, normalize=False, num_frames=duration * sample_rate)
        save_wav(path, data, sample_rate)

        waveform, sr = self._load(pathlib.Path(path))
        self.assertEqual(sr, sample_rate)
        self.assertEqual(waveform, data)

    @parameterized.expand(
        list(
            itertools.product(
                ["float32", "int32", "int16", "uint8"],
                [8000, 16000],
                [1, 2],
                [False, True],
            )
        ),
        name_func=name_func,
    )
    def test_wav(self, dtype, sample_rate, num_channels, normalize):
        """`self._load` can load wav format correctly."""
        self.assert_wav(dtype, sample_rate, num_channels, normalize, duration=1)

    @parameterized.expand(
        list(
            itertools.product(
                [8000, 16000],
                [1, 2],
                [False, True],
            )
        ),
        name_func=name_func,
    )
    def test_24bit_wav(self, sample_rate, num_channels, normalize):
        """`self._load` can load 24bit wav format correctly. Corectly casts it to ``int32`` tensor dtype."""
        self.assert_format("wav", sample_rate, num_channels, bit_depth=24, normalize=normalize, duration=1)

    @parameterized.expand(
        list(
            itertools.product(
                ["int16"],
                [16000],
                [2],
                [False],
            )
        ),
        name_func=name_func,
    )
    def test_wav_large(self, dtype, sample_rate, num_channels, normalize):
        """`self._load` can load large wav file correctly."""
        two_hours = 2 * 60 * 60
        self.assert_wav(dtype, sample_rate, num_channels, normalize, two_hours)

    @parameterized.expand(
        list(
            itertools.product(
                ["float32", "int32", "int16", "uint8"],
                [4, 8, 16],
            )
        ),
        name_func=name_func,
    )
    def test_multiple_channels(self, dtype, num_channels):
        """`self._load` can load wav file with more than 2 channels."""
        sample_rate = 8000
        normalize = False
        self.assert_wav(dtype, sample_rate, num_channels, normalize, duration=1)

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
        """`self._load` can load flac format correctly."""
        self.assert_format("flac", sample_rate, num_channels, compression=compression_level, bit_depth=16, duration=1)

    @parameterized.expand(
        list(
            itertools.product(
                [16000],
                [2],
                [0],
            )
        ),
        name_func=name_func,
    )
    def test_flac_large(self, sample_rate, num_channels, compression_level):
        """`self._load` can load large flac file correctly."""
        two_hours = 2 * 60 * 60
        self.assert_format(
            "flac", sample_rate, num_channels, compression=compression_level, bit_depth=16, duration=two_hours
        )

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
        """`self._load` can load vorbis format correctly."""
        self.assert_format("vorbis", sample_rate, num_channels, compression=quality_level, bit_depth=16, duration=1)

    @parameterized.expand(
        list(
            itertools.product(
                [16000],
                [2],
                [10],
            )
        ),
        name_func=name_func,
    )
    def test_vorbis_large(self, sample_rate, num_channels, quality_level):
        """`self._load` can load large vorbis file correctly."""
        two_hours = 2 * 60 * 60
        self.assert_format(
            "vorbis", sample_rate, num_channels, compression=quality_level, bit_depth=16, duration=two_hours
        )

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
        """`self._load` can load opus file correctly."""
        ops_path = get_asset_path("io", f"{bitrate}_{compression_level}_{num_channels}ch.opus")
        wav_path = self.get_temp_path(f"{bitrate}_{compression_level}_{num_channels}ch.opus.wav")
        _convert_audio_file(ops_path, wav_path, encoder="pcm_f32le")

        expected, sample_rate = load_wav(wav_path)
        found, sr = self._load(ops_path)

        assert sample_rate == sr
        self.assertEqual(expected, found)

    @parameterized.expand(
        list(
            itertools.product(
                [8000, 16000],
                [1, 2],
            )
        ),
        name_func=name_func,
    )
    def test_sphere(self, sample_rate, num_channels):
        """`self._load` can load sph format correctly."""
        self.assert_format("sph", sample_rate, num_channels, bit_depth=32, duration=1)

    @parameterized.expand(
        list(
            itertools.product(
                ["int16"],
                [3, 4, 16],
                [False, True],
            )
        ),
        name_func=name_func,
    )
    def test_amb(self, dtype, num_channels, normalize, sample_rate=8000):
        """`self._load` can load amb format correctly."""
        bit_depth = sox_utils.get_bit_depth(dtype)
        encoding = sox_utils.get_encoding(dtype)
        self.assert_format(
            "amb", sample_rate, num_channels, bit_depth=bit_depth, duration=1, encoding=encoding, normalize=normalize
        )

    # # NOTE: FFmpeg: RuntimeError: Failed to process a packet. (Not yet implemented in FFmpeg, patches welcome).
    # def test_amr_nb(self):
    #     """`self._load` can load amr_nb format correctly."""
    #     self.assert_format("amr-nb", sample_rate=8000, num_channels=1, bit_depth=32, duration=1)


@skipIfNoExec("sox")
@skipIfNoFFmpeg
class TestLoadWithoutExtension(PytorchTestCase):
    _load = partial(get_load_func(), backend="ffmpeg")

    def test_mp3(self):
        """MP3 file without extension can be loaded

        Originally, we added `format` argument for this case, but now we use FFmpeg
        for MP3 decoding, which works even without `format` argument.
        https://github.com/pytorch/audio/issues/1040

        The file was generated with the following command
            ffmpeg -f lavfi -i "sine=frequency=1000:duration=5" -ar 16000 -f mp3 test_noext
        """
        path = get_asset_path("mp3_without_ext")
        _, sr = self._load(path)
        assert sr == 16000

        with open(path, "rb") as fileobj:
            _, sr = self._load(fileobj)
        assert sr == 16000


class CloggedFileObj:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, _):
        return self.fileobj.read(2)

    def seek(self, offset, whence):
        return self.fileobj.seek(offset, whence)


@skipIfNoFFmpeg
@skipIfNoExec("sox")
class TestFileObject(TempDirMixin, PytorchTestCase):
    """
    In this test suite, the result of file-like object input is compared against file path input,
    because `load` function is rigrously tested for file path inputs to match libsox's result,
    """

    _load = partial(get_load_func(), backend="ffmpeg")

    @parameterized.expand(
        [
            ("wav", {"bit_depth": 16}),
            ("wav", {"bit_depth": 24}),
            ("wav", {"bit_depth": 32}),
            ("mp3", {"compression": 128}),
            ("mp3", {"compression": 320}),
            ("flac", {"compression": 0}),
            ("flac", {"compression": 5}),
            ("flac", {"compression": 8}),
            ("vorbis", {"compression": -1}),
            ("vorbis", {"compression": 10}),
            ("amb", {}),
        ]
    )
    def test_fileobj(self, ext, kwargs):
        """Loading audio via file object returns the same result as via file path."""
        sample_rate = 16000
        format_ = ext if ext in ["mp3"] else None
        path = self.get_temp_path(f"test.{ext}")

        sox_utils.gen_audio_file(path, sample_rate, num_channels=2, **kwargs)
        expected, _ = self._load(path)

        with open(path, "rb") as fileobj:
            found, sr = self._load(fileobj, format=format_)

        assert sr == sample_rate
        self.assertEqual(expected, found)

    @parameterized.expand(
        [
            ("wav", {"bit_depth": 16}),
            ("wav", {"bit_depth": 24}),
            ("wav", {"bit_depth": 32}),
            ("mp3", {"compression": 128}),
            ("mp3", {"compression": 320}),
            ("flac", {"compression": 0}),
            ("flac", {"compression": 5}),
            ("flac", {"compression": 8}),
            ("vorbis", {"compression": -1}),
            ("vorbis", {"compression": 10}),
            ("amb", {}),
        ]
    )
    def test_bytesio(self, ext, kwargs):
        """Loading audio via BytesIO object returns the same result as via file path."""
        sample_rate = 16000
        format_ = ext if ext in ["mp3"] else None
        path = self.get_temp_path(f"test.{ext}")

        sox_utils.gen_audio_file(path, sample_rate, num_channels=2, **kwargs)
        expected, _ = self._load(path)

        with open(path, "rb") as file_:
            fileobj = io.BytesIO(file_.read())
        found, sr = self._load(fileobj, format=format_)

        assert sr == sample_rate
        self.assertEqual(expected, found)

    @parameterized.expand(
        [
            ("wav", {"bit_depth": 16}),
            ("wav", {"bit_depth": 24}),
            ("wav", {"bit_depth": 32}),
            ("mp3", {"compression": 128}),
            ("mp3", {"compression": 320}),
            ("flac", {"compression": 0}),
            ("flac", {"compression": 5}),
            ("flac", {"compression": 8}),
            ("vorbis", {"compression": -1}),
            ("vorbis", {"compression": 10}),
            ("amb", {}),
        ]
    )
    def test_bytesio_clogged(self, ext, kwargs):
        """Loading audio via clogged file object returns the same result as via file path.

        This test case validates the case where fileobject returns shorter bytes than requeted.
        """
        sample_rate = 16000
        format_ = ext if ext in ["mp3"] else None
        path = self.get_temp_path(f"test.{ext}")

        sox_utils.gen_audio_file(path, sample_rate, num_channels=2, **kwargs)
        expected, _ = self._load(path)

        with open(path, "rb") as file_:
            fileobj = CloggedFileObj(io.BytesIO(file_.read()))
        found, sr = self._load(fileobj, format=format_)

        assert sr == sample_rate
        self.assertEqual(expected, found)

    @parameterized.expand(
        [
            ("wav", {"bit_depth": 16}),
            ("wav", {"bit_depth": 24}),
            ("wav", {"bit_depth": 32}),
            ("mp3", {"compression": 128}),
            ("mp3", {"compression": 320}),
            ("flac", {"compression": 0}),
            ("flac", {"compression": 5}),
            ("flac", {"compression": 8}),
            ("vorbis", {"compression": -1}),
            ("vorbis", {"compression": 10}),
            ("amb", {}),
        ]
    )
    def test_bytesio_tiny(self, ext, kwargs):
        """Loading very small audio via file object returns the same result as via file path."""
        sample_rate = 16000
        format_ = ext if ext in ["mp3"] else None
        path = self.get_temp_path(f"test.{ext}")

        sox_utils.gen_audio_file(path, sample_rate, num_channels=2, duration=1 / 1600, **kwargs)
        expected, _ = self._load(path)

        with open(path, "rb") as file_:
            fileobj = io.BytesIO(file_.read())
        found, sr = self._load(fileobj, format=format_)

        assert sr == sample_rate
        self.assertEqual(expected, found)

    @parameterized.expand(
        [
            ("wav", {"bit_depth": 16}),
            ("wav", {"bit_depth": 24}),
            ("wav", {"bit_depth": 32}),
            ("mp3", {"compression": 128}),
            ("mp3", {"compression": 320}),
            ("flac", {"compression": 0}),
            ("flac", {"compression": 5}),
            ("flac", {"compression": 8}),
            ("vorbis", {"compression": -1}),
            ("vorbis", {"compression": 10}),
            ("amb", {}),
        ]
    )
    def test_tarfile(self, ext, kwargs):
        """Loading compressed audio via file-like object returns the same result as via file path."""
        sample_rate = 16000
        format_ = ext if ext in ["mp3"] else None
        audio_file = f"test.{ext}"
        audio_path = self.get_temp_path(audio_file)
        archive_path = self.get_temp_path("archive.tar.gz")

        sox_utils.gen_audio_file(audio_path, sample_rate, num_channels=2, **kwargs)
        expected, _ = self._load(audio_path)

        with tarfile.TarFile(archive_path, "w") as tarobj:
            tarobj.add(audio_path, arcname=audio_file)
        with tarfile.TarFile(archive_path, "r") as tarobj:
            fileobj = tarobj.extractfile(audio_file)
            found, sr = self._load(fileobj, format=format_)

        assert sr == sample_rate
        self.assertEqual(expected, found)


class Unseekable:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, n):
        return self.fileobj.read(n)


@disabledInCI
@skipIfNoFFmpeg
@skipIfNoExec("sox")
@skipIfNoModule("requests")
class TestFileObjectHttp(HttpServerMixin, PytorchTestCase):
    _load = partial(get_load_func(), backend="ffmpeg")

    @parameterized.expand(
        [
            ("wav", {"bit_depth": 16}),
            ("wav", {"bit_depth": 24}),
            ("wav", {"bit_depth": 32}),
            ("mp3", {"compression": 128}),
            ("mp3", {"compression": 320}),
            ("flac", {"compression": 0}),
            ("flac", {"compression": 5}),
            ("flac", {"compression": 8}),
            ("vorbis", {"compression": -1}),
            ("vorbis", {"compression": 10}),
            ("amb", {}),
        ]
    )
    def test_requests(self, ext, kwargs):
        sample_rate = 16000
        format_ = ext if ext in ["mp3"] else None
        audio_file = f"test.{ext}"
        audio_path = self.get_temp_path(audio_file)

        sox_utils.gen_audio_file(audio_path, sample_rate, num_channels=2, **kwargs)
        expected, _ = self._load(audio_path)

        url = self.get_url(audio_file)
        with requests.get(url, stream=True) as resp:
            found, sr = self._load(Unseekable(resp.raw), format=format_)

        assert sr == sample_rate
        if ext != "mp3":
            self.assertEqual(expected, found)

    @parameterized.expand(
        list(
            itertools.product(
                [0, 1, 10, 100, 1000],
                [-1, 1, 10, 100, 1000],
            )
        ),
        name_func=name_func,
    )
    def test_frame(self, frame_offset, num_frames):
        """num_frames and frame_offset correctly specify the region of data"""
        sample_rate = 8000
        audio_file = "test.wav"
        audio_path = self.get_temp_path(audio_file)

        original = get_wav_data("float32", num_channels=2)
        save_wav(audio_path, original, sample_rate)
        frame_end = None if num_frames == -1 else frame_offset + num_frames
        expected = original[:, frame_offset:frame_end]

        url = self.get_url(audio_file)
        with requests.get(url, stream=True) as resp:
            found, sr = self._load(Unseekable(resp.raw), frame_offset, num_frames)

        assert sr == sample_rate
        self.assertEqual(expected, found)


@skipIfNoExec("sox")
@skipIfNoFFmpeg
class TestLoadNoSuchFile(PytorchTestCase):
    _load = partial(get_load_func(), backend="ffmpeg")

    def test_load_fail(self):
        """
        When attempted to load a non-existing file, error message must contain the file path.
        """
        path = "non_existing_audio.wav"
        with self.assertRaisesRegex(RuntimeError, path):
            self._load(path)
