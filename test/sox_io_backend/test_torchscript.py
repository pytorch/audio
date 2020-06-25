import itertools

import torch
from torchaudio.backend import sox_io_backend
from parameterized import parameterized

from ..common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    skipIfNoExec,
    skipIfNoExtension,
)
from .common import (
    get_test_name,
    get_wav_data,
    save_wav
)


def py_info_func(filepath: str) -> torch.classes.torchaudio.SignalInfo:
    return sox_io_backend.info(filepath)


def py_load_func(filepath: str, normalize: bool, channels_first: bool):
    return sox_io_backend.load(
        filepath, normalize=normalize, channels_first=channels_first)


@skipIfNoExec('sox')
@skipIfNoExtension
class SoxIO(TempDirMixin, TorchaudioTestCase):
    """TorchScript-ability Test suite for `sox_io_backend`"""
    @parameterized.expand(list(itertools.product(
        ['float32', 'int32', 'int16', 'uint8'],
        [8000, 16000],
        [1, 2],
    )), name_func=get_test_name)
    def test_info_wav(self, dtype, sample_rate, num_channels):
        """`sox_io_backend.info` is torchscript-able and returns the same result"""
        audio_path = self.get_temp_path(f'{dtype}_{sample_rate}_{num_channels}.wav')
        data = get_wav_data(dtype, num_channels, normalize=False, num_frames=1 * sample_rate)
        save_wav(audio_path, data, sample_rate)

        script_path = self.get_temp_path('info_func')
        torch.jit.script(py_info_func).save(script_path)
        ts_info_func = torch.jit.load(script_path)

        py_info = py_info_func(audio_path)
        ts_info = ts_info_func(audio_path)

        assert py_info.get_sample_rate() == ts_info.get_sample_rate()
        assert py_info.get_num_frames() == ts_info.get_num_frames()
        assert py_info.get_num_channels() == ts_info.get_num_channels()

    @parameterized.expand(list(itertools.product(
        ['float32', 'int32', 'int16', 'uint8'],
        [8000, 16000],
        [1, 2],
        [False, True],
        [False, True],
    )), name_func=get_test_name)
    def test_load_wav(self, dtype, sample_rate, num_channels, normalize, channels_first):
        """`sox_io_backend.load` is torchscript-able and returns the same result"""
        audio_path = self.get_temp_path(f'test_load_{dtype}_{sample_rate}_{num_channels}_{normalize}.wav')
        data = get_wav_data(dtype, num_channels, normalize=False, num_frames=1 * sample_rate)
        save_wav(audio_path, data, sample_rate)

        script_path = self.get_temp_path('load_func')
        torch.jit.script(py_load_func).save(script_path)
        ts_load_func = torch.jit.load(script_path)

        py_data, py_sr = py_load_func(
            audio_path, normalize=normalize, channels_first=channels_first)
        ts_data, ts_sr = ts_load_func(
            audio_path, normalize=normalize, channels_first=channels_first)

        self.assertEqual(py_sr, ts_sr)
        self.assertEqual(py_data, ts_data)
