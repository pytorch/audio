import itertools
from typing import Optional

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
    save_wav,
    load_wav,
)
from . import sox_utils


def py_info_func(filepath: str) -> torch.classes.torchaudio.SignalInfo:
    return sox_io_backend.info(filepath)


def py_load_func(filepath: str, normalize: bool, channels_first: bool):
    return sox_io_backend.load(
        filepath, normalize=normalize, channels_first=channels_first)


def py_save_func(
        filepath: str,
        tensor: torch.Tensor,
        sample_rate: int,
        channels_first: bool = True,
        compression: Optional[float] = None,
):
    sox_io_backend.save(filepath, tensor, sample_rate, channels_first, compression)


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

        script_path = self.get_temp_path('info_func.zip')
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

        script_path = self.get_temp_path('load_func.zip')
        torch.jit.script(py_load_func).save(script_path)
        ts_load_func = torch.jit.load(script_path)

        py_data, py_sr = py_load_func(
            audio_path, normalize=normalize, channels_first=channels_first)
        ts_data, ts_sr = ts_load_func(
            audio_path, normalize=normalize, channels_first=channels_first)

        self.assertEqual(py_sr, ts_sr)
        self.assertEqual(py_data, ts_data)

    @parameterized.expand(list(itertools.product(
        ['float32', 'int32', 'int16', 'uint8'],
        [8000, 16000],
        [1, 2],
    )), name_func=get_test_name)
    def test_save_wav(self, dtype, sample_rate, num_channels):
        script_path = self.get_temp_path('save_func.zip')
        torch.jit.script(py_save_func).save(script_path)
        ts_save_func = torch.jit.load(script_path)

        expected = get_wav_data(dtype, num_channels)
        py_path = self.get_temp_path(f'test_save_py_{dtype}_{sample_rate}_{num_channels}.wav')
        ts_path = self.get_temp_path(f'test_save_ts_{dtype}_{sample_rate}_{num_channels}.wav')

        py_save_func(py_path, expected, sample_rate, True, None)
        ts_save_func(ts_path, expected, sample_rate, True, None)

        py_data, py_sr = load_wav(py_path)
        ts_data, ts_sr = load_wav(ts_path)

        self.assertEqual(sample_rate, py_sr)
        self.assertEqual(sample_rate, ts_sr)
        self.assertEqual(expected, py_data)
        self.assertEqual(expected, ts_data)

    @parameterized.expand(list(itertools.product(
        [8000, 16000],
        [1, 2],
        list(range(9)),
    )), name_func=get_test_name)
    def test_save_flac(self, sample_rate, num_channels, compression_level):
        script_path = self.get_temp_path('save_func.zip')
        torch.jit.script(py_save_func).save(script_path)
        ts_save_func = torch.jit.load(script_path)

        expected = get_wav_data('float32', num_channels)
        py_path = self.get_temp_path(f'test_save_py_{sample_rate}_{num_channels}_{compression_level}.flac')
        ts_path = self.get_temp_path(f'test_save_ts_{sample_rate}_{num_channels}_{compression_level}.flac')

        py_save_func(py_path, expected, sample_rate, True, compression_level)
        ts_save_func(ts_path, expected, sample_rate, True, compression_level)

        # converting to 32 bit because flac file has 24 bit depth which scipy cannot handle.
        py_path_wav = f'{py_path}.wav'
        ts_path_wav = f'{ts_path}.wav'
        sox_utils.convert_audio_file(py_path, py_path_wav, bit_depth=32)
        sox_utils.convert_audio_file(ts_path, ts_path_wav, bit_depth=32)

        py_data, py_sr = load_wav(py_path_wav, normalize=True)
        ts_data, ts_sr = load_wav(ts_path_wav, normalize=True)

        self.assertEqual(sample_rate, py_sr)
        self.assertEqual(sample_rate, ts_sr)
        self.assertEqual(expected, py_data)
        self.assertEqual(expected, ts_data)
