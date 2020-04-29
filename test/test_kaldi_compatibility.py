"""Test suites for checking numerical compatibility against Kaldi"""
import shutil
import unittest
import subprocess

import kaldi_io
import torch
import torchaudio.functional as F


def _exe_exists(cmd):
    return shutil.which(cmd) is not None


def _convert_args(**kwargs):
    args = []
    for key, value in kwargs.items():
        key = '--' + key.replace('_', '-')
        value = str(value).lower() if value in [True, False] else str(value)
        args.append('%s=%s' % (key, value))
    return args


def _run_kaldi(command, input_tensor):
    """Run provided Kaldi command, pass a tensor and get the resulting tensor

    Assumption:
        The provided Kaldi command consumes one ark and produces one ark.
        i.e. 'ark:- ark:-'
    """
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    kaldi_io.write_mat(process.stdin, input_tensor.numpy(), key='foo')
    process.stdin.close()
    result = dict(kaldi_io.read_mat_ark(process.stdout))['foo']
    return torch.from_numpy(result.copy())  # copy supresses some torch warning


class TestFunctional:
    @unittest.skipUnless(_exe_exists('apply-cmvn-sliding'), '`apply-cmvn-sliding` not available')
    def test_sliding_window_cmn(self):
        """sliding_window_cmn should be numerically compatible with apply-cmvn-sliding"""
        kwargs = {
            'cmn_window': 600,
            'min_cmn_window': 100,
            'center': False,
            'norm_vars': False,
        }

        tensor = torch.randn(40, 10)
        result = F.sliding_window_cmn(tensor, **kwargs)
        command = ['apply-cmvn-sliding'] + _convert_args(**kwargs) + ['ark:-', 'ark:-']
        kaldi_result = _run_kaldi(command, tensor)
        torch.testing.assert_allclose(result, kaldi_result)
