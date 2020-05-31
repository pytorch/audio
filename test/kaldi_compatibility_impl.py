"""Test suites for checking numerical compatibility against Kaldi"""
import shutil
import unittest
import subprocess

import kaldi_io
import torch
import torchaudio.functional as F
import torchaudio.compliance.kaldi

from . import common_utils


def _not_available(cmd):
    return shutil.which(cmd) is None


def _convert_args(**kwargs):
    args = []
    for key, value in kwargs.items():
        key = '--' + key.replace('_', '-')
        value = str(value).lower() if value in [True, False] else str(value)
        args.append('%s=%s' % (key, value))
    return args


def _get_func_args(test_name, keys, args_str):
    """Return the python dict of arguments for a input arg string , based on test_name

    Arguments:
        test_name: Name of the test ( Ex: 'fbank' , 'mfcc' )
        keys : Arugument keys corresponding to the test
        args_str : string of arguments joined by `-`
    """
    args_dict = {}
    args = args_str.split('-')

    for i in range(len(args) - 1):
        args_dict[keys[i]] = _parse(args[i + 1])

    if test_name in ('fbank', 'mfcc'):
        args_dict['dither'] = 0.0

    return args_dict

TEST_PREFIX = ['spec', 'fbank', 'mfcc', 'resample']


def _parse(token):
    """converts an string argument(token) to its corresponding python type

    Arguments:
        token: string
    """
    if token == 'true':
        return True
    if token == 'false':
        return False
    if token in torchaudio.compliance.kaldi.WINDOWS or token in [TEST_PREFIX]:
        return token
    if '.' in token:
        return float(token)
    return int(token)


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

    @unittest.skipIf(_not_available('compute-fbank-feats'), '`compute-fbank-feats` not available')
    def test_fbank(self):
        """fbank should be numerically compatible with compute-fbank-feats"""

        wave_file = common_utils.get_asset_path('kaldi_file.wav')
        waveform = torchaudio.load_wav(wave_file)[0].to(dtype=self.dtype, device=self.device)

        kaldi_arg_file = common_utils.get_asset_path('kaldi_test_args.txt')
        args_list = [line.strip() for line in open(kaldi_arg_file, "r")]
        fbank_args_list = [args for args in args_list if 'fbank' in args]

        fbank_keys = ['blackman_coeff',
                      'energy_floor',
                      'frame_length',
                      'frame_shift',
                      'high_freq',
                      'htk_compat',
                      'low_freq',
                      'num_mel_bins',
                      'preemphasis_coefficient',
                      'raw_energy',
                      'remove_dc_offset',
                      'round_to_power_of_two',
                      'snip_edges',
                      'subtract_mean',
                      'use_energy',
                      'use_log_fbank',
                      'use_power',
                      'vtln_high',
                      'vtln_low',
                      'vtln_warp',
                      'window_type',
                      'dither',
                      ]

        for args_string in fbank_args_list:
            kwargs = _get_func_args('fbank', fbank_keys, args_string)
            result = torchaudio.compliance.kaldi.fbank(waveform, **kwargs)
            command = ['compute-fbank-feats'] + _convert_args(**kwargs) + ['scp:-', 'ark:-']
            kaldi_result = _run_kaldi(command, 'scp', wave_file)
            self.assert_equal(result, expected=kaldi_result, rtol=1e-4, atol=1e-8)
