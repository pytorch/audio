import itertools

from torchaudio.backend import sox_io_backend
from parameterized import parameterized

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    PytorchTestCase,
    skipIfNoExec,
    skipIfNoExtension,
    get_wav_data,
)
from .common import (
    name_func,
)


@skipIfNoExec('sox')
@skipIfNoExtension
class TestRoundTripIO(TempDirMixin, PytorchTestCase):
    """save/load round trip should not degrade data for lossless formats"""
    @parameterized.expand(list(itertools.product(
        ['float32', 'int32', 'int16', 'uint8'],
        [8000, 16000],
        [1, 2],
    )), name_func=name_func)
    def test_wav(self, dtype, sample_rate, num_channels):
        """save/load round trip should not degrade data for wav formats"""
        original = get_wav_data(dtype, num_channels, normalize=False)
        data = original
        for i in range(10):
            path = self.get_temp_path(f'{i}.wav')
            sox_io_backend.save(path, data, sample_rate)
            data, sr = sox_io_backend.load(path, normalize=False)
            assert sr == sample_rate
            self.assertEqual(original, data)

    @parameterized.expand(list(itertools.product(
        [8000, 16000],
        [1, 2],
        list(range(9)),
    )), name_func=name_func)
    def test_flac(self, sample_rate, num_channels, compression_level):
        """save/load round trip should not degrade data for flac formats"""
        original = get_wav_data('float32', num_channels)
        data = original
        for i in range(10):
            path = self.get_temp_path(f'{i}.flac')
            sox_io_backend.save(path, data, sample_rate, compression=compression_level)
            data, sr = sox_io_backend.load(path)
            assert sr == sample_rate
            self.assertEqual(original, data)
