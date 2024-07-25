import itertools
from typing import Optional

import torch
import torchaudio
from parameterized import parameterized
from torchaudio_unittest.common_utils import (
    get_wav_data,
    load_wav,
    save_wav,
    skipIfNoExec,
    skipIfNoSox,
    sox_utils,
    TempDirMixin,
    torch_script,
    TorchaudioTestCase,
)

from .common import get_enc_params, name_func


def py_info_func(filepath: str) -> torchaudio.backend.sox_io_backend.AudioMetaData:
    return torchaudio.backend.sox_io_backend.info(filepath)


def py_load_func(filepath: str, normalize: bool, channels_first: bool):
    return torchaudio.backend.sox_io_backend.load(filepath, normalize=normalize, channels_first=channels_first)


def py_save_func(
    filepath: str,
    tensor: torch.Tensor,
    sample_rate: int,
    channels_first: bool = True,
    compression: Optional[float] = None,
    encoding: Optional[str] = None,
    bits_per_sample: Optional[int] = None,
):
    torchaudio.backend.sox_io_backend.save(
        filepath, tensor, sample_rate, channels_first, compression, None, encoding, bits_per_sample
    )


@skipIfNoExec("sox")
@skipIfNoSox
class SoxIO(TempDirMixin, TorchaudioTestCase):
    """TorchScript-ability Test suite for `sox_io_backend`"""

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
    def test_info_wav(self, dtype, sample_rate, num_channels):
        """`sox_io_backend.info` is torchscript-able and returns the same result"""
        audio_path = self.get_temp_path(f"{dtype}_{sample_rate}_{num_channels}.wav")
        data = get_wav_data(dtype, num_channels, normalize=False, num_frames=1 * sample_rate)
        save_wav(audio_path, data, sample_rate)

        ts_info_func = torch_script(py_info_func)

        py_info = py_info_func(audio_path)
        ts_info = ts_info_func(audio_path)

        assert py_info.sample_rate == ts_info.sample_rate
        assert py_info.num_frames == ts_info.num_frames
        assert py_info.num_channels == ts_info.num_channels

    @parameterized.expand(
        list(
            itertools.product(
                ["float32", "int32", "int16", "uint8"],
                [8000, 16000],
                [1, 2],
                [False, True],
                [False, True],
            )
        ),
        name_func=name_func,
    )
    def test_load_wav(self, dtype, sample_rate, num_channels, normalize, channels_first):
        """`sox_io_backend.load` is torchscript-able and returns the same result"""
        audio_path = self.get_temp_path(f"test_load_{dtype}_{sample_rate}_{num_channels}_{normalize}.wav")
        data = get_wav_data(dtype, num_channels, normalize=False, num_frames=1 * sample_rate)
        save_wav(audio_path, data, sample_rate)

        ts_load_func = torch_script(py_load_func)

        py_data, py_sr = py_load_func(audio_path, normalize=normalize, channels_first=channels_first)
        ts_data, ts_sr = ts_load_func(audio_path, normalize=normalize, channels_first=channels_first)

        self.assertEqual(py_sr, ts_sr)
        self.assertEqual(py_data, ts_data)

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
    def test_save_wav(self, dtype, sample_rate, num_channels):
        ts_save_func = torch_script(py_save_func)

        expected = get_wav_data(dtype, num_channels, normalize=False)
        py_path = self.get_temp_path(f"test_save_py_{dtype}_{sample_rate}_{num_channels}.wav")
        ts_path = self.get_temp_path(f"test_save_ts_{dtype}_{sample_rate}_{num_channels}.wav")
        enc, bps = get_enc_params(dtype)

        py_save_func(py_path, expected, sample_rate, True, None, enc, bps)
        ts_save_func(ts_path, expected, sample_rate, True, None, enc, bps)

        py_data, py_sr = load_wav(py_path, normalize=False)
        ts_data, ts_sr = load_wav(ts_path, normalize=False)

        self.assertEqual(sample_rate, py_sr)
        self.assertEqual(sample_rate, ts_sr)
        self.assertEqual(expected, py_data)
        self.assertEqual(expected, ts_data)

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
    def test_save_flac(self, sample_rate, num_channels, compression_level):
        ts_save_func = torch_script(py_save_func)

        expected = get_wav_data("float32", num_channels)
        py_path = self.get_temp_path(f"test_save_py_{sample_rate}_{num_channels}_{compression_level}.flac")
        ts_path = self.get_temp_path(f"test_save_ts_{sample_rate}_{num_channels}_{compression_level}.flac")

        py_save_func(py_path, expected, sample_rate, True, compression_level, None, None)
        ts_save_func(ts_path, expected, sample_rate, True, compression_level, None, None)

        # converting to 32 bit because flac file has 24 bit depth which scipy cannot handle.
        py_path_wav = f"{py_path}.wav"
        ts_path_wav = f"{ts_path}.wav"
        sox_utils.convert_audio_file(py_path, py_path_wav, bit_depth=32)
        sox_utils.convert_audio_file(ts_path, ts_path_wav, bit_depth=32)

        py_data, py_sr = load_wav(py_path_wav, normalize=True)
        ts_data, ts_sr = load_wav(ts_path_wav, normalize=True)

        self.assertEqual(sample_rate, py_sr)
        self.assertEqual(sample_rate, ts_sr)
        self.assertEqual(expected, py_data)
        self.assertEqual(expected, ts_data)
