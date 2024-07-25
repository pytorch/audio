import itertools

import torch
from parameterized import parameterized
from torchaudio.backend import sox_io_backend
from torchaudio_unittest.common_utils import (
    get_asset_path,
    get_wav_data,
    load_wav,
    nested_params,
    PytorchTestCase,
    save_wav,
    skipIfNoExec,
    skipIfNoSox,
    skipIfNoSoxDecoder,
    sox_utils,
    TempDirMixin,
)

from .common import name_func


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
    @skipIfNoSoxDecoder("opus")
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

    @skipIfNoSoxDecoder("amr-nb")
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

        self._test(sox_io_backend.load, frame_offset, num_frames, channels_first, normalize)


@skipIfNoSox
class TestLoadWithoutExtension(PytorchTestCase):
    def test_mp3(self):
        """MP3 file without extension can be loaded

        https://github.com/pytorch/audio/issues/1040

        The file was generated with the following command
            ffmpeg -f lavfi -i "sine=frequency=1000:duration=5" -ar 16000 -f mp3 test_noext
        """
        path = get_asset_path("mp3_without_ext")
        _, sr = sox_io_backend.load(path, format="mp3")
        assert sr == 16000


@skipIfNoSox
class TestLoadNoSuchFile(PytorchTestCase):
    def test_load_fail(self):
        """
        When attempted to load a non-existing file, error message must contain the file path.
        """
        path = "non_existing_audio.wav"
        with self.assertRaisesRegex(RuntimeError, path):
            sox_io_backend.load(path)
