"""Test suites for torchscriptability of SoX I/O and its comatibility with Python"""
from typing import Tuple

import torch
from torch.testing._internal.common_utils import TestCase
import torchaudio

import common_utils

SignalInfo = torchaudio._soundfile_backend.SignalInfo
EncodingInfo = torchaudio._soundfile_backend.EncodingInfo


class TestSoxIO(TestCase):
    """Implements test for `sox` C++ extensions"""
    def test_get_info(self):
        def func(path: str) -> Tuple[SignalInfo, EncodingInfo]:
            return torchaudio._sox_backend.info(path)

        path = common_utils.get_asset_path('sinewave.wav')
        ts_func = torch.jit.script(func)
        py_signal_info, py_encoding_info = func(path)
        ts_signal_info, ts_encoding_info = ts_func(path)

        assert py_signal_info.__dict__ == ts_signal_info.__dict__
        assert py_encoding_info.__dict__ == ts_encoding_info.__dict__
