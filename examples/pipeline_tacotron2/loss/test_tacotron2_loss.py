import os
import unittest
import tempfile

import torch
from torch.autograd import gradcheck, gradgradcheck

from loss_function import Tacotron2Loss


def skipIfNoCuda(test_item):
    if torch.cuda.is_available():
        return test_item
    force_cuda_test = os.environ.get("TORCHAUDIO_TEST_FORCE_CUDA", "0")
    if force_cuda_test not in ["0", "1"]:
        raise ValueError('"TORCHAUDIO_TEST_FORCE_CUDA" must be either "0" or "1".')
    if force_cuda_test == "1":
        raise RuntimeError(
            '"TORCHAUDIO_TEST_FORCE_CUDA" is set but CUDA is not available.'
        )
    return unittest.skip("CUDA is not available.")(test_item)


class TempDirMixin:
    """Mixin to provide easy access to temp dir"""

    temp_dir_ = None

    @classmethod
    def get_base_temp_dir(cls):
        # If TORCHAUDIO_TEST_TEMP_DIR is set, use it instead of temporary directory.
        # this is handy for debugging.
        key = "TORCHAUDIO_TEST_TEMP_DIR"
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


def _get_inputs(dtype, device):
    n_mel, n_batch, max_mel_specgram_length = 3, 2, 4
    mel_specgram = torch.rand(
        n_batch, n_mel, max_mel_specgram_length, dtype=dtype, device=device
    )
    mel_specgram_postnet = torch.rand(
        n_batch, n_mel, max_mel_specgram_length, dtype=dtype, device=device
    )
    gate_out = torch.rand(n_batch, dtype=dtype, device=device)
    truth_mel_specgram = torch.rand(
        n_batch, n_mel, max_mel_specgram_length, dtype=dtype, device=device
    )
    truth_gate_out = torch.rand(n_batch, dtype=dtype, device=device)

    return (
        mel_specgram,
        mel_specgram_postnet,
        gate_out,
        truth_mel_specgram,
        truth_gate_out,
    )


class Tacotron2LossTest(unittest.TestCase, TempDirMixin):

    dtype = torch.float64
    device = "cpu"

    def _assert_torchscript_consistency(self, fn, tensors):
        path = self.get_temp_path("func.zip")
        torch.jit.script(fn).save(path)
        ts_func = torch.jit.load(path)

        torch.random.manual_seed(40)
        output = fn(*tensors)

        torch.random.manual_seed(40)
        ts_output = ts_func(*tensors)

        self.assertEqual(ts_output, output)

    def test_cpu_torchscript_consistency(self):
        f"""Validate the torchscript consistency of Tacotron2Loss."""
        dtype = torch.float32
        device = torch.device("cpu")

        def _fn(mel_specgram, mel_specgram_postnet, gate_out, truth_mel_specgram, truth_gate_out):
            loss_fn = Tacotron2Loss()
            return loss_fn(
                (mel_specgram, mel_specgram_postnet, gate_out),
                (truth_mel_specgram, truth_gate_out),
            )

        self._assert_torchscript_consistency(_fn, _get_inputs(dtype, device))

    @skipIfNoCuda
    def test_gpu_torchscript_consistency(self):
        f"""Validate the torchscript consistency of Tacotron2Loss."""
        dtype = torch.float32
        device = torch.device("cuda")

        def _fn(mel_specgram, mel_specgram_postnet, gate_out, truth_mel_specgram, truth_gate_out):
            loss_fn = Tacotron2Loss()
            return loss_fn(
                (mel_specgram, mel_specgram_postnet, gate_out),
                (truth_mel_specgram, truth_gate_out),
            )

        self._assert_torchscript_consistency(_fn, self._get_inputs(dtype, device))

    def test_cpu_gradcheck(self):
        f"""Performing gradient check on Tacotron2Loss."""
        dtype = torch.float64  # gradcheck needs a higher numerical accuracy
        device = torch.device("cuda")

        (
            mel_specgram,
            mel_specgram_postnet,
            gate_out,
            truth_mel_specgram,
            truth_gate_out,
        ) = _get_inputs(dtype, device)

        mel_specgram.requires_grad_(True)
        mel_specgram_postnet.requires_grad_(True)
        gate_out.requires_grad_(True)

        def _fn(mel_specgram, mel_specgram_postnet, gate_out, truth_mel_specgram, truth_gate_out):
            loss_fn = Tacotron2Loss()
            return loss_fn(
                (mel_specgram, mel_specgram_postnet, gate_out),
                (truth_mel_specgram, truth_gate_out),
            )

        gradcheck(
            _fn,
            (mel_specgram, mel_specgram_postnet, gate_out, truth_mel_specgram, truth_gate_out),
            fast_mode=True,
        )
        gradgradcheck(
            _fn,
            (mel_specgram, mel_specgram_postnet, gate_out, truth_mel_specgram, truth_gate_out),
            fast_mode=True,
        )

    @skipIfNoCuda
    def test_gpu_gradcheck(self):
        f"""Performing gradient check on Tacotron2Loss."""
        dtype = torch.float64  # gradcheck needs a higher numerical accuracy
        device = torch.device("cuda")

        (
            mel_specgram,
            mel_specgram_postnet,
            gate_out,
            truth_mel_specgram,
            truth_gate_out,
        ) = _get_inputs(dtype, device)

        mel_specgram.requires_grad_(True)
        mel_specgram_postnet.requires_grad_(True)
        gate_out.requires_grad_(True)

        def _fn(mel_specgram, mel_specgram_postnet, gate_out, truth_mel_specgram, truth_gate_out):
            loss_fn = Tacotron2Loss()
            return loss_fn(
                (mel_specgram, mel_specgram_postnet, gate_out),
                (truth_mel_specgram, truth_gate_out),
            )

        gradcheck(
            _fn,
            (mel_specgram, mel_specgram_postnet, gate_out, truth_mel_specgram, truth_gate_out),
            fast_mode=True,
        )
        gradgradcheck(
            _fn,
            (mel_specgram, mel_specgram_postnet, gate_out, truth_mel_specgram, truth_gate_out),
            fast_mode=True,
        )


if __name__ == "__main__":
    unittest.main()
