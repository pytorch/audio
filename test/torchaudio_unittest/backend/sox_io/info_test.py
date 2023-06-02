import itertools

from parameterized import parameterized
from torchaudio.backend import sox_io_backend
from torchaudio_unittest.backend.common import get_encoding
from torchaudio_unittest.common_utils import (
    get_asset_path,
    get_wav_data,
    PytorchTestCase,
    save_wav,
    skipIfNoExec,
    skipIfNoSox,
    skipIfNoSoxDecoder,
    sox_utils,
    TempDirMixin,
)

from .common import name_func


@skipIfNoExec("sox")
@skipIfNoSox
class TestInfo(TempDirMixin, PytorchTestCase):
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
        """`sox_io_backend.info` can check wav file correctly"""
        duration = 1
        path = self.get_temp_path("data.wav")
        data = get_wav_data(dtype, num_channels, normalize=False, num_frames=duration * sample_rate)
        save_wav(path, data, sample_rate)
        info = sox_io_backend.info(path)
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
        """`sox_io_backend.info` can check wav file with channels more than 2 correctly"""
        duration = 1
        path = self.get_temp_path("data.wav")
        data = get_wav_data(dtype, num_channels, normalize=False, num_frames=duration * sample_rate)
        save_wav(path, data, sample_rate)
        info = sox_io_backend.info(path)
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
                [96, 128, 160, 192, 224, 256, 320],
            )
        ),
        name_func=name_func,
    )
    def test_mp3(self, sample_rate, num_channels, bit_rate):
        """`sox_io_backend.info` can check mp3 file correctly"""
        duration = 1
        path = self.get_temp_path("data.mp3")
        sox_utils.gen_audio_file(
            path,
            sample_rate,
            num_channels,
            compression=bit_rate,
            duration=duration,
        )
        info = sox_io_backend.info(path)
        assert info.sample_rate == sample_rate
        # mp3 does not preserve the number of samples
        # assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels
        assert info.bits_per_sample == 0  # bit_per_sample is irrelevant for compressed formats
        assert info.encoding == "MP3"

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
        """`sox_io_backend.info` can check flac file correctly"""
        duration = 1
        path = self.get_temp_path("data.flac")
        sox_utils.gen_audio_file(
            path,
            sample_rate,
            num_channels,
            compression=compression_level,
            duration=duration,
        )
        info = sox_io_backend.info(path)
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
        """`sox_io_backend.info` can check vorbis file correctly"""
        duration = 1
        path = self.get_temp_path("data.vorbis")
        sox_utils.gen_audio_file(
            path,
            sample_rate,
            num_channels,
            compression=quality_level,
            duration=duration,
        )
        info = sox_io_backend.info(path)
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
        """`sox_io_backend.info` can check sph file correctly"""
        duration = 1
        path = self.get_temp_path("data.sph")
        sox_utils.gen_audio_file(path, sample_rate, num_channels, duration=duration, bit_depth=bits_per_sample)
        info = sox_io_backend.info(path)
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
        """`sox_io_backend.info` can check amb file correctly"""
        duration = 1
        path = self.get_temp_path("data.amb")
        bits_per_sample = sox_utils.get_bit_depth(dtype)
        sox_utils.gen_audio_file(path, sample_rate, num_channels, bit_depth=bits_per_sample, duration=duration)
        info = sox_io_backend.info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels
        assert info.bits_per_sample == bits_per_sample
        assert info.encoding == get_encoding("amb", dtype)

    @skipIfNoSoxDecoder("amr_nb")
    def test_amr_nb(self):
        """`sox_io_backend.info` can check amr-nb file correctly"""
        duration = 1
        num_channels = 1
        sample_rate = 8000
        path = self.get_temp_path("data.amr-nb")
        sox_utils.gen_audio_file(
            path, sample_rate=sample_rate, num_channels=num_channels, bit_depth=16, duration=duration
        )
        info = sox_io_backend.info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels
        assert info.bits_per_sample == 0
        assert info.encoding == "AMR_NB"

    def test_ulaw(self):
        """`sox_io_backend.info` can check ulaw file correctly"""
        duration = 1
        num_channels = 1
        sample_rate = 8000
        path = self.get_temp_path("data.wav")
        sox_utils.gen_audio_file(
            path, sample_rate=sample_rate, num_channels=num_channels, bit_depth=8, encoding="u-law", duration=duration
        )
        info = sox_io_backend.info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels
        assert info.bits_per_sample == 8
        assert info.encoding == "ULAW"

    def test_alaw(self):
        """`sox_io_backend.info` can check alaw file correctly"""
        duration = 1
        num_channels = 1
        sample_rate = 8000
        path = self.get_temp_path("data.wav")
        sox_utils.gen_audio_file(
            path, sample_rate=sample_rate, num_channels=num_channels, bit_depth=8, encoding="a-law", duration=duration
        )
        info = sox_io_backend.info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels
        assert info.bits_per_sample == 8
        assert info.encoding == "ALAW"

    def test_gsm(self):
        """`sox_io_backend.info` can check gsm file correctly"""
        duration = 1
        num_channels = 1
        sample_rate = 8000
        path = self.get_temp_path("data.gsm")
        sox_utils.gen_audio_file(path, sample_rate=sample_rate, num_channels=num_channels, duration=duration)
        info = sox_io_backend.info(path)
        assert info.sample_rate == sample_rate
        assert info.num_channels == num_channels
        assert info.bits_per_sample == 0
        assert info.encoding == "GSM"

    def test_htk(self):
        """`sox_io_backend.info` can check HTK file correctly"""
        duration = 1
        num_channels = 1
        sample_rate = 8000
        path = self.get_temp_path("data.htk")
        sox_utils.gen_audio_file(
            path, sample_rate=sample_rate, num_channels=num_channels, bit_depth=16, duration=duration
        )
        info = sox_io_backend.info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels
        assert info.bits_per_sample == 16
        assert info.encoding == "PCM_S"


@skipIfNoSox
class TestInfoOpus(PytorchTestCase):
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
        """`sox_io_backend.info` can check opus file correcty"""
        path = get_asset_path("io", f"{bitrate}_{compression_level}_{num_channels}ch.opus")
        info = sox_io_backend.info(path)
        assert info.sample_rate == 48000
        assert info.num_frames == 32768
        assert info.num_channels == num_channels
        assert info.bits_per_sample == 0  # bit_per_sample is irrelevant for compressed formats
        assert info.encoding == "OPUS"


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
        sinfo = sox_io_backend.info(path)
        assert sinfo.sample_rate == 16000
        assert sinfo.num_frames == 80000
        assert sinfo.num_channels == 1
        assert sinfo.bits_per_sample == 0  # bit_per_sample is irrelevant for compressed formats
        assert sinfo.encoding == "MP3"


@skipIfNoSox
class TestInfoNoSuchFile(PytorchTestCase):
    def test_info_fail(self):
        """
        When attempted to get info on a non-existing file, error message must contain the file path.
        """
        path = "non_existing_audio.wav"
        with self.assertRaisesRegex(RuntimeError, path):
            sox_io_backend.info(path)
