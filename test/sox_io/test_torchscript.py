import itertools

import torch
from torchaudio.backend import sox_io_backend
from parameterized import parameterized

from .. import common_utils
from ..common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
)
from .common import (
    get_test_name,
)
from . import sox_utils


def py_info_func(filepath: str) -> torch.classes.torchaudio.SignalInfo:
    return sox_io_backend.info(filepath)


@common_utils.skipIfNoExec('sox')
@common_utils.skipIfNoExtension
class SoxIO(TempDirMixin, TorchaudioTestCase):
    @parameterized.expand(list(itertools.product(
        ['float32', 'int32', 'int16', 'uint8'],
        [8000, 16000],
        [1, 2],
    )), name_func=get_test_name)
    def test_info_wav(self, dtype, sample_rate, num_channels):
        audio_path = self.get_temp_path(f'{dtype}_{sample_rate}_{num_channels}.wav')
        sox_utils.gen_audio_file(
            audio_path, sample_rate, num_channels,
            bit_depth=sox_utils.get_bit_depth(dtype),
            encoding=sox_utils.get_encoding(dtype),
        )

        script_path = self.get_temp_path('info_func')
        torch.jit.script(py_info_func).save(script_path)
        ts_info_func = torch.jit.load(script_path)

        py_info = py_info_func(audio_path)
        ts_info = ts_info_func(audio_path)

        assert py_info.get_sample_rate() == ts_info.get_sample_rate()
        assert py_info.get_num_samples() == ts_info.get_num_samples()
        assert py_info.get_num_channels() == ts_info.get_num_channels()
