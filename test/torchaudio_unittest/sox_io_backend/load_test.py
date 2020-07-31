import itertools

from torchaudio.backend import sox_io_backend
from parameterized import parameterized

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    PytorchTestCase,
    skipIfNoExec,
    skipIfNoExtension,
    get_asset_path,
    get_wav_data,
    load_wav,
    save_wav,
    sox_utils,
)
from .common import (
    name_func,
)


class LoadTestBase(TempDirMixin, PytorchTestCase):
    def assert_wav(self, dtype, sample_rate, num_channels, normalize, duration):
        """`sox_io_backend.load` can load wav format correctly.

        Wav data loaded with sox_io backend should match those with scipy
        """
        path = self.get_temp_path('reference.wav')
        data = get_wav_data(dtype, num_channels, normalize=normalize, num_frames=duration * sample_rate)
        save_wav(path, data, sample_rate)
        expected = load_wav(path, normalize=normalize)[0]
        data, sr = sox_io_backend.load(path, normalize=normalize)
        assert sr == sample_rate
        self.assertEqual(data, expected)

    def assert_mp3(self, sample_rate, num_channels, bit_rate, duration):
        """`sox_io_backend.load` can load mp3 format.

        mp3 encoding introduces delay and boundary effects so
        we create reference wav file from mp3

         x
         |
         | 1. Generate mp3 with Sox
         |
         v    2. Convert to wav with Sox
        mp3 ------------------------------> wav
         |                                   |
         | 3. Load with torchaudio           | 4. Load with scipy
         |                                   |
         v                                   v
        tensor ----------> x <----------- tensor
                       5. Compare

        Underlying assumptions are;
        i. Conversion of mp3 to wav with Sox preserves data.
        ii. Loading wav file with scipy is correct.

        By combining i & ii, step 2. and 4. allows to load reference mp3 data
        without using torchaudio
        """
        path = self.get_temp_path('1.original.mp3')
        ref_path = self.get_temp_path('2.reference.wav')

        # 1. Generate mp3 with sox
        sox_utils.gen_audio_file(
            path, sample_rate, num_channels,
            compression=bit_rate, duration=duration)
        # 2. Convert to wav with sox
        sox_utils.convert_audio_file(path, ref_path)
        # 3. Load mp3 with torchaudio
        data, sr = sox_io_backend.load(path)
        # 4. Load wav with scipy
        data_ref = load_wav(ref_path)[0]
        # 5. Compare
        assert sr == sample_rate
        self.assertEqual(data, data_ref, atol=3e-03, rtol=1.3e-06)

    def assert_flac(self, sample_rate, num_channels, compression_level, duration):
        """`sox_io_backend.load` can load flac format.

        This test takes the same strategy as mp3 to compare the result
        """
        path = self.get_temp_path('1.original.flac')
        ref_path = self.get_temp_path('2.reference.wav')

        # 1. Generate flac with sox
        sox_utils.gen_audio_file(
            path, sample_rate, num_channels,
            compression=compression_level, bit_depth=16, duration=duration)
        # 2. Convert to wav with sox
        sox_utils.convert_audio_file(path, ref_path)
        # 3. Load flac with torchaudio
        data, sr = sox_io_backend.load(path)
        # 4. Load wav with scipy
        data_ref = load_wav(ref_path)[0]
        # 5. Compare
        assert sr == sample_rate
        self.assertEqual(data, data_ref, atol=4e-05, rtol=1.3e-06)

    def assert_vorbis(self, sample_rate, num_channels, quality_level, duration):
        """`sox_io_backend.load` can load vorbis format.

        This test takes the same strategy as mp3 to compare the result
        """
        path = self.get_temp_path('1.original.vorbis')
        ref_path = self.get_temp_path('2.reference.wav')

        # 1. Generate vorbis with sox
        sox_utils.gen_audio_file(
            path, sample_rate, num_channels,
            compression=quality_level, bit_depth=16, duration=duration)
        # 2. Convert to wav with sox
        sox_utils.convert_audio_file(path, ref_path)
        # 3. Load vorbis with torchaudio
        data, sr = sox_io_backend.load(path)
        # 4. Load wav with scipy
        data_ref = load_wav(ref_path)[0]
        # 5. Compare
        assert sr == sample_rate
        self.assertEqual(data, data_ref, atol=4e-05, rtol=1.3e-06)


@skipIfNoExec('sox')
@skipIfNoExtension
class TestLoad(LoadTestBase):
    """Test the correctness of `sox_io_backend.load` for various formats"""
    @parameterized.expand(list(itertools.product(
        ['float32', 'int32', 'int16', 'uint8'],
        [8000, 16000],
        [1, 2],
        [False, True],
    )), name_func=name_func)
    def test_wav(self, dtype, sample_rate, num_channels, normalize):
        """`sox_io_backend.load` can load wav format correctly."""
        self.assert_wav(dtype, sample_rate, num_channels, normalize, duration=1)

    @parameterized.expand(list(itertools.product(
        ['int16'],
        [16000],
        [2],
        [False],
    )), name_func=name_func)
    def test_wav_large(self, dtype, sample_rate, num_channels, normalize):
        """`sox_io_backend.load` can load large wav file correctly."""
        two_hours = 2 * 60 * 60
        self.assert_wav(dtype, sample_rate, num_channels, normalize, two_hours)

    @parameterized.expand(list(itertools.product(
        ['float32', 'int32', 'int16', 'uint8'],
        [4, 8, 16, 32],
    )), name_func=name_func)
    def test_multiple_channels(self, dtype, num_channels):
        """`sox_io_backend.load` can load wav file with more than 2 channels."""
        sample_rate = 8000
        normalize = False
        self.assert_wav(dtype, sample_rate, num_channels, normalize, duration=1)

    @parameterized.expand(list(itertools.product(
        [8000, 16000, 44100],
        [1, 2],
        [96, 128, 160, 192, 224, 256, 320],
    )), name_func=name_func)
    def test_mp3(self, sample_rate, num_channels, bit_rate):
        """`sox_io_backend.load` can load mp3 format correctly."""
        self.assert_mp3(sample_rate, num_channels, bit_rate, duration=1)

    @parameterized.expand(list(itertools.product(
        [16000],
        [2],
        [128],
    )), name_func=name_func)
    def test_mp3_large(self, sample_rate, num_channels, bit_rate):
        """`sox_io_backend.load` can load large mp3 file correctly."""
        two_hours = 2 * 60 * 60
        self.assert_mp3(sample_rate, num_channels, bit_rate, two_hours)

    @parameterized.expand(list(itertools.product(
        [8000, 16000],
        [1, 2],
        list(range(9)),
    )), name_func=name_func)
    def test_flac(self, sample_rate, num_channels, compression_level):
        """`sox_io_backend.load` can load flac format correctly."""
        self.assert_flac(sample_rate, num_channels, compression_level, duration=1)

    @parameterized.expand(list(itertools.product(
        [16000],
        [2],
        [0],
    )), name_func=name_func)
    def test_flac_large(self, sample_rate, num_channels, compression_level):
        """`sox_io_backend.load` can load large flac file correctly."""
        two_hours = 2 * 60 * 60
        self.assert_flac(sample_rate, num_channels, compression_level, two_hours)

    @parameterized.expand(list(itertools.product(
        [8000, 16000],
        [1, 2],
        [-1, 0, 1, 2, 3, 3.6, 5, 10],
    )), name_func=name_func)
    def test_vorbis(self, sample_rate, num_channels, quality_level):
        """`sox_io_backend.load` can load vorbis format correctly."""
        self.assert_vorbis(sample_rate, num_channels, quality_level, duration=1)

    @parameterized.expand(list(itertools.product(
        [16000],
        [2],
        [10],
    )), name_func=name_func)
    def test_vorbis_large(self, sample_rate, num_channels, quality_level):
        """`sox_io_backend.load` can load large vorbis file correctly."""
        two_hours = 2 * 60 * 60
        self.assert_vorbis(sample_rate, num_channels, quality_level, two_hours)

    @parameterized.expand(list(itertools.product(
        ['96k'],
        [1, 2],
        [0, 5, 10],
    )), name_func=name_func)
    def test_opus(self, bitrate, num_channels, compression_level):
        """`sox_io_backend.load` can load opus file correctly."""
        ops_path = get_asset_path('io', f'{bitrate}_{compression_level}_{num_channels}ch.opus')
        wav_path = self.get_temp_path(f'{bitrate}_{compression_level}_{num_channels}ch.opus.wav')
        sox_utils.convert_audio_file(ops_path, wav_path)

        expected, sample_rate = load_wav(wav_path)
        found, sr = sox_io_backend.load(ops_path)

        assert sample_rate == sr
        self.assertEqual(expected, found)


@skipIfNoExec('sox')
@skipIfNoExtension
class TestLoadParams(TempDirMixin, PytorchTestCase):
    """Test the correctness of frame parameters of `sox_io_backend.load`"""
    original = None
    path = None

    def setUp(self):
        super().setUp()
        sample_rate = 8000
        self.original = get_wav_data('float32', num_channels=2)
        self.path = self.get_temp_path('test.wave')
        save_wav(self.path, self.original, sample_rate)

    @parameterized.expand(list(itertools.product(
        [0, 1, 10, 100, 1000],
        [-1, 1, 10, 100, 1000],
    )), name_func=name_func)
    def test_frame(self, frame_offset, num_frames):
        """num_frames and frame_offset correctly specify the region of data"""
        found, _ = sox_io_backend.load(self.path, frame_offset, num_frames)
        frame_end = None if num_frames == -1 else frame_offset + num_frames
        self.assertEqual(found, self.original[:, frame_offset:frame_end])

    @parameterized.expand([(True, ), (False, )], name_func=name_func)
    def test_channels_first(self, channels_first):
        """channels_first swaps axes"""
        found, _ = sox_io_backend.load(self.path, channels_first=channels_first)
        expected = self.original if channels_first else self.original.transpose(1, 0)
        self.assertEqual(found, expected)
