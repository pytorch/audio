import os
import unittest
import tempfile

import torch
from torch.autograd import gradcheck, gradgradcheck

from loss_function import Tacotron2Loss


class TempDirMixin:
    """Mixin to provide easy access to temp dir"""
    temp_dir_ = None

    @classmethod
    def get_base_temp_dir(cls):
        # If TORCHAUDIO_TEST_TEMP_DIR is set, use it instead of temporary directory.
        # this is handy for debugging.
        key = 'TORCHAUDIO_TEST_TEMP_DIR'
        if key in os.environ:
            return os.environ[key]
        if cls.temp_dir_ is None:
            cls.temp_dir_ = tempfile.TemporaryDirectory()
        return cls.temp_dir_.name

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if cls.temp_dir_ is not None:
            cls.temp_dir_.cleanup()
            cls.temp_dir_ = None

    def get_temp_path(self, *paths):
        temp_dir = os.path.join(self.get_base_temp_dir(), self.id())
        path = os.path.join(temp_dir, *paths)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path


class Tacotron2LossTest(unittest.TestCase, TempDirMixin):

    dtype = torch.float64
    device = "cpu"

    def _assert_torchscript_consistency(self, fn, tensors):
        path = self.get_temp_path('func.zip')
        torch.jit.script(fn).save(path)
        ts_func = torch.jit.load(path)

        torch.random.manual_seed(40)
        output = fn(*tensors)

        torch.random.manual_seed(40)
        ts_output = ts_func(*tensors)

        self.assertEqual(ts_output, output)

    def _get_inputs(self):
        n_mel, n_batch, max_mel_specgram_length = 5, 2, 8
        mel_specgram = torch.rand(n_batch, n_mel, max_mel_specgram_length, dtype=self.dtype, device=self.device)
        mel_specgram_postnet = torch.rand(n_batch, n_mel, max_mel_specgram_length, dtype=self.dtype, device=self.device)
        gate_out = torch.rand(n_batch, dtype=self.dtype, device=self.device)
        truth_mel_specgram = torch.rand(n_batch, n_mel, max_mel_specgram_length, dtype=self.dtype, device=self.device)
        truth_gate_out = torch.rand(n_batch, dtype=self.dtype, device=self.device)

        return mel_specgram, mel_specgram_postnet, gate_out, truth_mel_specgram, truth_gate_out

    def test_torchscript_consistency(self):
        f"""Validate the torchscript consistency of Tacotron2Loss."""

        def _fn(mel_specgram, mel_specgram_postnet, gate_out, truth_mel_specgram, truth_gate_out):
            loss_fn = Tacotron2Loss()
            return loss_fn((mel_specgram, mel_specgram_postnet, gate_out), (truth_mel_specgram, truth_gate_out))

        self._assert_torchscript_consistency(_fn, self._get_inputs())

    def test_gradcheck(self):
        f"""Performing gradient check on Tacotron2Loss."""

        mel_specgram, mel_specgram_postnet, gate_out, truth_mel_specgram, truth_gate_out = self._get_inputs()

        mel_specgram.requires_grad_(True)
        mel_specgram_postnet.requires_grad_(True)
        gate_out.requires_grad_(True)

        def _fn(mel_specgram, mel_specgram_postnet, gate_out, truth_mel_specgram, truth_gate_out):
            loss_fn = Tacotron2Loss()
            return loss_fn((mel_specgram, mel_specgram_postnet, gate_out), (truth_mel_specgram, truth_gate_out))

        gradcheck(_fn, (mel_specgram, mel_specgram_postnet, gate_out, truth_mel_specgram, truth_gate_out), fast_mode=True)
        gradgradcheck(_fn, (mel_specgram, mel_specgram_postnet, gate_out, truth_mel_specgram, truth_gate_out), fast_mode=True)

if __name__ == "__main__":
    unittest.main()
