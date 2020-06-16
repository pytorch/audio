import itertools
from parameterized import parameterized

from torchaudio.backend import sox_io_backend

from .. import common_utils
from ..common_utils import (
    TempDirMixin,
    PytorchTestCase,
)
from .common import (
    get_test_name
)
from . import sox_utils


@common_utils.skipIfNoExec('sox')
@common_utils.skipIfNoExtension
class TestInfo(TempDirMixin, PytorchTestCase):
    @parameterized.expand(list(itertools.product(
        ['float32', 'int32', 'int16', 'uint8'],
        [8000, 16000],
        [1, 2],
    )), name_func=get_test_name)
    def test_info_wav(self, dtype, sample_rate, num_channels):
        path = self.get_temp_path(f'{dtype}_{sample_rate}_{num_channels}.wav')
        sox_utils.gen_audio_file(
            path, sample_rate, num_channels,
            bit_depth=sox_utils.get_bit_depth(dtype),
            encoding=sox_utils.get_encoding(dtype),
        )
        info = sox_io_backend.info(path)
        assert info.get_sample_rate() == sample_rate
        assert info.get_num_samples() == sample_rate
        assert info.get_num_channels() == num_channels

    @parameterized.expand(list(itertools.product(
        ['float32', 'int32', 'int16', 'uint8'],
        [8000, 16000],
        [4, 8, 16, 32],
    )), name_func=get_test_name)
    def test_info_wav_multiple_channels(self, dtype, sample_rate, num_channels):
        """`sox_io_backend.save` can save wav with more than 2 channels."""
        path = self.get_temp_path(f'{dtype}_{sample_rate}_{num_channels}.wav')
        sox_utils.gen_audio_file(
            path, sample_rate, num_channels,
            bit_depth=sox_utils.get_bit_depth(dtype),
            encoding=sox_utils.get_encoding(dtype),
        )
        info = sox_io_backend.info(path)
        assert info.get_sample_rate() == sample_rate
        assert info.get_num_samples() == sample_rate
        assert info.get_num_channels() == num_channels

    @parameterized.expand(list(itertools.product(
        [8000, 16000],
        [1, 2],
        [96, 128, 160, 192, 224, 256, 320],
    )), name_func=get_test_name)
    def test_info_mp3(self, sample_rate, num_channels, bit_rate):
        path = self.get_temp_path(f'{sample_rate}_{num_channels}_{bit_rate}k.mp3')
        sox_utils.gen_audio_file(path, sample_rate, num_channels, compression=bit_rate)
        info = sox_io_backend.info(path)
        assert info.get_sample_rate() == sample_rate
        # assert info.get_num_samples() == sample_rate
        assert info.get_num_channels() == num_channels

    @parameterized.expand(list(itertools.product(
        [8000, 16000],
        [1, 2],
        list(range(9)),
    )), name_func=get_test_name)
    def test_info_flac(self, sample_rate, num_channels, compression_level):
        path = self.get_temp_path(f'{sample_rate}_{num_channels}_{compression_level}.flac')
        sox_utils.gen_audio_file(path, sample_rate, num_channels, compression=compression_level)
        info = sox_io_backend.info(path)
        assert info.get_sample_rate() == sample_rate
        assert info.get_num_samples() == sample_rate
        assert info.get_num_channels() == num_channels

    @parameterized.expand(list(itertools.product(
        [8000, 16000],
        [1, 2],
        [-1, 0, 1, 2, 3, 3.6, 5, 10],
    )), name_func=get_test_name)
    def test_info_vorbis(self, sample_rate, num_channels, quality_level):
        path = self.get_temp_path(f'{sample_rate}_{num_channels}_{quality_level}.vorbis')
        sox_utils.gen_audio_file(path, sample_rate, num_channels, compression=quality_level)
        info = sox_io_backend.info(path)
        assert info.get_sample_rate() == sample_rate
        assert info.get_num_samples() == sample_rate
        assert info.get_num_channels() == num_channels
