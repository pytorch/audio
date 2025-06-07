import itertools
from functools import partial

from parameterized import parameterized
from torchaudio._backend.utils import get_load_func, get_save_func
from torchaudio_unittest.common_utils import get_wav_data, PytorchTestCase, skipIfNoExec, skipIfNoSox, TempDirMixin

from .common import get_enc_params, name_func


@skipIfNoExec("sox")
@skipIfNoSox
class TestRoundTripIO(TempDirMixin, PytorchTestCase):
    """save/load round trip should not degrade data for lossless formats"""

    _load = staticmethod(partial(get_load_func(), backend="sox"))
    _save = staticmethod(partial(get_save_func(), backend="sox"))

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
        """save/load round trip should not degrade data for wav formats"""
        original = get_wav_data(dtype, num_channels, normalize=False)
        enc, bps = get_enc_params(dtype)
        data = original
        for i in range(10):
            path = self.get_temp_path(f"{i}.wav")
            self._save(path, data, sample_rate, encoding=enc, bits_per_sample=bps)
            data, sr = self._load(path, normalize=False)
            assert sr == sample_rate
            self.assertEqual(original, data)

    @parameterized.expand(
        list(
            itertools.product(
                [8000, 16000],
                [1, 2],
            )
        ),
        name_func=name_func,
    )
    def test_flac(self, sample_rate, num_channels):
        """save/load round trip should not degrade data for flac formats"""
        original = get_wav_data("float32", num_channels)
        data = original
        for i in range(10):
            path = self.get_temp_path(f"{i}.flac")
            self._save(path, data, sample_rate)
            data, sr = self._load(path)
            assert sr == sample_rate
            self.assertEqual(original, data)
