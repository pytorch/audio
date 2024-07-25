import itertools
from functools import partial

from parameterized import parameterized
from torchaudio._backend.utils import get_info_func, get_load_func, get_save_func
from torchaudio_unittest.common_utils import get_wav_data, skipIfNoSox, TempDirMixin, TorchaudioTestCase

from .common import name_func


@skipIfNoSox
class SmokeTest(TempDirMixin, TorchaudioTestCase):
    """Run smoke test on various audio format

    The purpose of this test suite is to verify that sox_io_backend functionalities do not exhibit
    abnormal behaviors.

    This test suite should be able to run without any additional tools (such as sox command),
    however without such tools, the correctness of each function cannot be verified.
    """

    _info = partial(get_info_func(), backend="sox")
    _load = partial(get_load_func(), backend="sox")
    _save = partial(get_save_func(), backend="sox")

    def run_smoke_test(self, ext, sample_rate, num_channels, *, dtype="float32"):
        duration = 1
        num_frames = sample_rate * duration
        path = self.get_temp_path(f"test.{ext}")
        original = get_wav_data(dtype, num_channels, normalize=False, num_frames=num_frames)

        # 1. run save
        self._save(path, original, sample_rate)
        # 2. run info
        info = self._info(path)
        assert info.sample_rate == sample_rate
        assert info.num_channels == num_channels
        # 3. run load
        loaded, sr = self._load(path, normalize=False)
        assert sr == sample_rate
        assert loaded.shape[0] == num_channels

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
        """Run smoke test on wav format"""
        self.run_smoke_test("wav", sample_rate, num_channels, dtype=dtype)

    @parameterized.expand(
        list(
            itertools.product(
                [8000, 16000],
                [1, 2],
            )
        )
    )
    def test_vorbis(self, sample_rate, num_channels):
        """Run smoke test on vorbis format"""
        self.run_smoke_test("vorbis", sample_rate, num_channels)

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
        """Run smoke test on flac format"""
        self.run_smoke_test("flac", sample_rate, num_channels)
