import io
import itertools
import tarfile

import torch
import torchaudio
from parameterized import parameterized
from torchaudio._internal import module_utils as _mod_utils
from torchaudio.backend import sox_io_backend
from torchaudio_unittest.common_utils import (
    get_asset_path,
    get_wav_data,
    HttpServerMixin,
    load_wav,
    nested_params,
    PytorchTestCase,
    save_wav,
    skipIfNoExec,
    skipIfNoModule,
    skipIfNoSox,
    sox_utils,
    TempDirMixin,
)

from .common import name_func


if _mod_utils.is_module_available("requests"):
    import requests


class LoadTestBase(TempDirMixin, PytorchTestCase):
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
        """`sox_io_backend.load` can load given format correctly.

        file encodings introduce delay and boundary effects so
        we create a reference wav file from the original file format

         x
         |
         |    1. Generate given format with Sox
         |
         v    2. Convert to wav with Sox
        given format ----------------------> wav
         |                                   |
         |    3. Load with torchaudio        | 4. Load with scipy
         |                                   |
         v                                   v
        tensor ----------> x <----------- tensor
                       5. Compare

        Underlying assumptions are;
        i. Conversion of given format to wav with Sox preserves data.
        ii. Loading wav file with scipy is correct.

        By combining i & ii, step 2. and 4. allows to load reference given format
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
        # 2. Convert to wav with sox
        wav_bit_depth = 32 if bit_depth == 24 else None  # for 24-bit wav
        sox_utils.convert_audio_file(path, ref_path, bit_depth=wav_bit_depth)
        # 3. Load the given format with torchaudio
        data, sr = sox_io_backend.load(path, normalize=normalize)
        # 4. Load wav with scipy
        data_ref = load_wav(ref_path, normalize=normalize)[0]
        # 5. Compare
        assert sr == sample_rate
        self.assertEqual(data, data_ref, atol=atol, rtol=rtol)

    def assert_wav(self, dtype, sample_rate, num_channels, normalize, duration):
        """`sox_io_backend.load` can load wav format correctly.

        Wav data loaded with sox_io backend should match those with scipy
        """
        path = self.get_temp_path("reference.wav")
        data = get_wav_data(dtype, num_channels, normalize=normalize, num_frames=duration * sample_rate)
        save_wav(path, data, sample_rate)
        expected = load_wav(path, normalize=normalize)[0]
        data, sr = sox_io_backend.load(path, normalize=normalize)
        assert sr == sample_rate
        self.assertEqual(data, expected)


@skipIfNoExec("sox")
@skipIfNoSox
class TestLoad(LoadTestBase):
    """Test the correctness of `sox_io_backend.load` for various formats"""

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
        """`sox_io_backend.load` can load wav format correctly."""
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
        """`sox_io_backend.load` can load 24bit wav format correctly. Corectly casts it to ``int32`` tensor dtype."""
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
        """`sox_io_backend.load` can load large wav file correctly."""
        two_hours = 2 * 60 * 60
        self.assert_wav(dtype, sample_rate, num_channels, normalize, two_hours)

    @parameterized.expand(
        list(
            itertools.product(
                ["float32", "int32", "int16", "uint8"],
                [4, 8, 16, 32],
            )
        ),
        name_func=name_func,
    )
    def test_multiple_channels(self, dtype, num_channels):
        """`sox_io_backend.load` can load wav file with more than 2 channels."""
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
        """`sox_io_backend.load` can load flac format correctly."""
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
        """`sox_io_backend.load` can load large flac file correctly."""
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
        """`sox_io_backend.load` can load vorbis format correctly."""
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
        """`sox_io_backend.load` can load large vorbis file correctly."""
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
        """`sox_io_backend.load` can load opus file correctly."""
        ops_path = get_asset_path("io", f"{bitrate}_{compression_level}_{num_channels}ch.opus")
        wav_path = self.get_temp_path(f"{bitrate}_{compression_level}_{num_channels}ch.opus.wav")
        sox_utils.convert_audio_file(ops_path, wav_path)

        expected, sample_rate = load_wav(wav_path)
        found, sr = sox_io_backend.load(ops_path)

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
        """`sox_io_backend.load` can load sph format correctly."""
        self.assert_format("sph", sample_rate, num_channels, bit_depth=32, duration=1)

    @parameterized.expand(
        list(
            itertools.product(
                ["float32", "int32", "int16"],
                [8000, 16000],
                [1, 2],
                [False, True],
            )
        ),
        name_func=name_func,
    )
    def test_amb(self, dtype, sample_rate, num_channels, normalize):
        """`sox_io_backend.load` can load amb format correctly."""
        bit_depth = sox_utils.get_bit_depth(dtype)
        encoding = sox_utils.get_encoding(dtype)
        self.assert_format(
            "amb", sample_rate, num_channels, bit_depth=bit_depth, duration=1, encoding=encoding, normalize=normalize
        )

    def test_amr_nb(self):
        """`sox_io_backend.load` can load amr_nb format correctly."""
        self.assert_format("amr-nb", sample_rate=8000, num_channels=1, bit_depth=32, duration=1)


@skipIfNoSox
class TestLoadParams(TempDirMixin, PytorchTestCase):
    """Test the correctness of frame parameters of `sox_io_backend.load`"""

    def _test(self, func, frame_offset, num_frames, channels_first, normalize):
        original = get_wav_data("int16", num_channels=2, normalize=False)
        path = self.get_temp_path("test.wav")
        save_wav(path, original, sample_rate=8000)

        output, _ = func(path, frame_offset, num_frames, normalize, channels_first, None)
        frame_end = None if num_frames == -1 else frame_offset + num_frames
        expected = original[:, slice(frame_offset, frame_end)]
        if not channels_first:
            expected = expected.T
        if normalize:
            expected = expected.to(torch.float32) / (2**15)
        self.assertEqual(output, expected)

    @nested_params(
        [0, 1, 10, 100, 1000],
        [-1, 1, 10, 100, 1000],
        [True, False],
        [True, False],
    )
    def test_sox(self, frame_offset, num_frames, channels_first, normalize):
        """The combination of properly changes the output tensor"""

        self._test(torch.ops.torchaudio.sox_io_load_audio_file, frame_offset, num_frames, channels_first, normalize)

        # test file-like obj
        def func(path, *args):
            with open(path, "rb") as fileobj:
                return torchaudio.lib._torchaudio_sox.load_audio_fileobj(fileobj, *args)

        self._test(func, frame_offset, num_frames, channels_first, normalize)

    @nested_params(
        [0, 1, 10, 100, 1000],
        [-1, 1, 10, 100, 1000],
        [True, False],
        [True, False],
    )
    def test_ffmpeg(self, frame_offset, num_frames, channels_first, normalize):
        """The combination of properly changes the output tensor"""
        from torchaudio.io._compat import load_audio, load_audio_fileobj

        self._test(load_audio, frame_offset, num_frames, channels_first, normalize)

        # test file-like obj
        def func(path, *args):
            with open(path, "rb") as fileobj:
                return load_audio_fileobj(fileobj, *args)

        self._test(func, frame_offset, num_frames, channels_first, normalize)


@skipIfNoSox
class TestLoadWithoutExtension(PytorchTestCase):
    def test_mp3(self):
        """MP3 file without extension can be loaded

        Originally, we added `format` argument for this case, but now we use FFmpeg
        for MP3 decoding, which works even without `format` argument.
        https://github.com/pytorch/audio/issues/1040

        The file was generated with the following command
            ffmpeg -f lavfi -i "sine=frequency=1000:duration=5" -ar 16000 -f mp3 test_noext
        """
        path = get_asset_path("mp3_without_ext")
        _, sr = sox_io_backend.load(path)
        assert sr == 16000

        with open(path, "rb") as fileobj:
            _, sr = sox_io_backend.load(fileobj)
        assert sr == 16000


class CloggedFileObj:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, _):
        return self.fileobj.read(2)

    def seek(self, offset, whence):
        return self.fileobj.seek(offset, whence)


@skipIfNoSox
@skipIfNoExec("sox")
class TestFileObject(TempDirMixin, PytorchTestCase):
    """
    In this test suite, the result of file-like object input is compared against file path input,
    because `load` function is rigrously tested for file path inputs to match libsox's result,
    """

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
        expected, _ = sox_io_backend.load(path)

        with open(path, "rb") as fileobj:
            found, sr = sox_io_backend.load(fileobj, format=format_)

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
        expected, _ = sox_io_backend.load(path)

        with open(path, "rb") as file_:
            fileobj = io.BytesIO(file_.read())
        found, sr = sox_io_backend.load(fileobj, format=format_)

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
        expected, _ = sox_io_backend.load(path)

        with open(path, "rb") as file_:
            fileobj = CloggedFileObj(io.BytesIO(file_.read()))
        found, sr = sox_io_backend.load(fileobj, format=format_)

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
        expected, _ = sox_io_backend.load(path)

        with open(path, "rb") as file_:
            fileobj = io.BytesIO(file_.read())
        found, sr = sox_io_backend.load(fileobj, format=format_)

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
        expected, _ = sox_io_backend.load(audio_path)

        with tarfile.TarFile(archive_path, "w") as tarobj:
            tarobj.add(audio_path, arcname=audio_file)
        with tarfile.TarFile(archive_path, "r") as tarobj:
            fileobj = tarobj.extractfile(audio_file)
            found, sr = sox_io_backend.load(fileobj, format=format_)

        assert sr == sample_rate
        self.assertEqual(expected, found)


class Unseekable:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, n):
        return self.fileobj.read(n)


@skipIfNoSox
@skipIfNoExec("sox")
@skipIfNoModule("requests")
class TestFileObjectHttp(HttpServerMixin, PytorchTestCase):
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
        expected, _ = sox_io_backend.load(audio_path)

        url = self.get_url(audio_file)
        with requests.get(url, stream=True) as resp:
            found, sr = sox_io_backend.load(Unseekable(resp.raw), format=format_)

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
            found, sr = sox_io_backend.load(resp.raw, frame_offset, num_frames)

        assert sr == sample_rate
        self.assertEqual(expected, found)


@skipIfNoSox
class TestLoadNoSuchFile(PytorchTestCase):
    def test_load_fail(self):
        """
        When attempted to load a non-existing file, error message must contain the file path.
        """
        path = "non_existing_audio.wav"
        with self.assertRaisesRegex(RuntimeError, path):
            sox_io_backend.load(path)
