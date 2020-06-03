"""Test suites for checking numerical compatibility against Kaldi"""
import json
import shutil
import unittest
import subprocess

import kaldi_io
import torch
import torchaudio.functional as F
import torchaudio.compliance.kaldi

from . import common_utils
from parameterized import parameterized, param


def _not_available(cmd):
    return shutil.which(cmd) is None


def _convert_args(**kwargs):
    args = []
    for key, value in kwargs.items():
        key = '--' + key.replace('_', '-')
        value = str(value).lower() if value in [True, False] else str(value)
        args.append('%s=%s' % (key, value))
    return args


def _run_kaldi(command, input_type, input_value):
    """Run provided Kaldi command, pass a tensor and get the resulting tensor

    Arguments:
        input_type: str
            'ark' or 'scp'
        input_value:
            Tensor for 'ark'
            string for 'scp' (path to an audio file)
    """
    key = 'foo'
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    if input_type == 'ark':
        kaldi_io.write_mat(process.stdin, input_value.cpu().numpy(), key=key)
    elif input_type == 'scp':
        process.stdin.write(f'{key} {input_value}'.encode('utf8'))
    else:
        raise NotImplementedError('Unexpected type')
    process.stdin.close()
    result = dict(kaldi_io.read_mat_ark(process.stdout))['foo']
    return torch.from_numpy(result.copy())  # copy supresses some torch warning


def _load_params(path):
    with open(path, 'r') as file:
        return [param(json.loads(line)) for line in file]


class Kaldi(common_utils.TestBaseMixin):
    def assert_equal(self, output, *, expected, rtol=None, atol=None):
        expected = expected.to(dtype=self.dtype, device=self.device)
        self.assertEqual(output, expected, rtol=rtol, atol=atol)

    @unittest.skipIf(_not_available('apply-cmvn-sliding'), '`apply-cmvn-sliding` not available')
    def test_sliding_window_cmn(self):
        """sliding_window_cmn should be numerically compatible with apply-cmvn-sliding"""
        kwargs = {
            'cmn_window': 600,
            'min_cmn_window': 100,
            'center': False,
            'norm_vars': False,
        }

        tensor = torch.randn(40, 10, dtype=self.dtype, device=self.device)
        result = F.sliding_window_cmn(tensor, **kwargs)
        command = ['apply-cmvn-sliding'] + _convert_args(**kwargs) + ['ark:-', 'ark:-']
        kaldi_result = _run_kaldi(command, 'ark', tensor)
        self.assert_equal(result, expected=kaldi_result)

    @parameterized.expand(_load_params(common_utils.get_asset_path('kaldi_test_fbank_args.json')))
    @unittest.skipIf(_not_available('compute-fbank-feats'), '`compute-fbank-feats` not available')
    def test_fbank(self, kwargs):
        """fbank should be numerically compatible with compute-fbank-feats"""
        wave_file = common_utils.get_asset_path('kaldi_file.wav')
        waveform = torchaudio.load_wav(wave_file)[0].to(dtype=self.dtype, device=self.device)
        result = torchaudio.compliance.kaldi.fbank(waveform, **kwargs)
        command = ['compute-fbank-feats'] + _convert_args(**kwargs) + ['scp:-', 'ark:-']
        kaldi_result = _run_kaldi(command, 'scp', wave_file)
        self.assert_equal(result, expected=kaldi_result, rtol=1e-4, atol=1e-8)

    @parameterized.expand(_load_params(common_utils.get_asset_path('kaldi_test_mfcc_args.json')))
    @unittest.skipIf(_not_available('compute-mfcc-feats'), '`compute-mfcc-feats` not available')
    def test_mfcc(self, kwargs):
        """mfcc should be numerically compatible with compute-mfcc-feats"""
        wave_file = common_utils.get_asset_path('kaldi_file.wav')
        waveform = torchaudio.load_wav(wave_file)[0].to(dtype=self.dtype, device=self.device)
        result = torchaudio.compliance.kaldi.mfcc(waveform, **kwargs)
        command = ['compute-mfcc-feats'] + _convert_args(**kwargs) + ['scp:-', 'ark:-']
        kaldi_result = _run_kaldi(command, 'scp', wave_file)
        self.assert_equal(result, expected=kaldi_result, rtol=1e-4, atol=1e-8)
