"""Test suites for checking numerical compatibility against Kaldi"""
import shutil
import unittest
import subprocess

import kaldi_io
import torch
import torchaudio.functional as F
import torchaudio.compliance.kaldi as kaldi
import torchaudio
import math

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
    test_8000_filepath = common_utils.get_asset_path('kaldi_file_8000.wav')

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

    @unittest.skipIf(_not_available('compute-mfcc-feats'), '`compute-mfcc-feats` not available')
    def test_mfcc(self):
        """mfcc should be numerically compatible with compute-mfcc-feats"""

        wave_file = common_utils.get_asset_path('kaldi_file.wav')
        waveform = torchaudio.load_wav(wave_file)[0].to(dtype=self.dtype, device=self.device)

        kaldi_arg_file = common_utils.get_asset_path('kaldi_test_args.txt')
        args_list = [line.strip() for line in open(kaldi_arg_file, "r")]
        mfcc_args_list = [args for args in args_list if 'mfcc' in args]

        mfcc_keys = ['blackman_coeff',
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
                     'num_ceps',
                     'cepstral_lifter',
                     'vtln_high',
                     'vtln_low',
                     'vtln_warp',
                     'window_type',
                     'dither',
                     ]

        for args_string in mfcc_args_list:
            kwargs = _get_func_args('mfcc', mfcc_keys, args_string)
            result = torchaudio.compliance.kaldi.mfcc(waveform, **kwargs)
            command = ['compute-mfcc-feats'] + _convert_args(**kwargs) + ['scp:-', 'ark:-']
            kaldi_result = _run_kaldi(command, 'scp', wave_file)
            self.assert_equal(result, expected=kaldi_result, rtol=1e-4, atol=1e-8)

    @unittest.skipIf(_not_available('compute-spectrogram-feats'), '`compute-spectrogram-feats` not available')
    def test_spectrogram(self):
        """spectrogram should be numerically compatible with compute-spectrogram-feats"""

        wave_file = common_utils.get_asset_path('kaldi_file.wav')
        waveform = torchaudio.load_wav(wave_file)[0].to(dtype=self.dtype, device=self.device)

        kaldi_arg_file = common_utils.get_asset_path('kaldi_test_args.txt')
        args_list = [line.strip() for line in open(kaldi_arg_file, "r")]
        spec_args_list = [args for args in args_list if 'spec' in args]

        spec_keys = ['blackman_coeff',
                     'dither',
                     'energy_floor',
                     'frame_length',
                     'frame_shift',
                     'preemphasis_coefficient',
                     'raw_energy',
                     'remove_dc_offset',
                     'round_to_power_of_two',
                     'snip_edges',
                     'subtract_mean',
                     'window_type',
                     ]

        for args_string in spec_args_list:
            kwargs = _get_func_args('spec', spec_keys, args_string)
            result = torchaudio.compliance.kaldi.spectrogram(waveform, **kwargs)
            command = ['compute-spectrogram-feats'] + _convert_args(**kwargs) + ['scp:-', 'ark:-']
            kaldi_result = _run_kaldi(command, 'scp', wave_file)
            self.assert_equal(result, expected=kaldi_result, rtol=1e-4, atol=1e-8)

    def test_mfcc_empty(self):
        # Passing in an empty tensor should result in an error
        self.assertRaises(AssertionError, kaldi.mfcc, torch.empty(0))

    def test_resample(self):
        wave_file = self.test_8000_filepath
        waveform = torchaudio.load_wav(wave_file)[0].to(dtype=self.dtype, device=self.device)

        kaldi_arg_file = common_utils.get_asset_path('kaldi_test_args.txt')
        args_list = [line.strip() for line in open(kaldi_arg_file, "r")]
        spec_args_list = [args for args in args_list if 'resample' in args]

        resample_keys = ['orig_freq',
                         'new_freq',
                         ]

        for args_string in spec_args_list:
            kwargs = _get_func_args('resample', resample_keys, args_string)
            result = torchaudio.compliance.kaldi.spectrogram(waveform, **kwargs)
            # TODO: kaldi command for resample ??
            # command = ['compute-spectrogram-feats'] + _convert_args(**kwargs) + ['scp:-', 'ark:-']
            # kaldi_result = _run_kaldi(command, 'scp', wave_file)
            kaldi_result = None
            self.assert_equal(result, expected=kaldi_result, rtol=1e-4, atol=1e-8)

    def test_resample_waveform_upsample_size(self):
        sound, sample_rate = torchaudio.load_wav(self.test_8000_filepath)
        upsample_sound = kaldi.resample_waveform(sound, sample_rate, sample_rate * 2)
        self.assertTrue(upsample_sound.size(-1) == sound.size(-1) * 2)

    def test_resample_waveform_downsample_size(self):
        sound, sample_rate = torchaudio.load_wav(self.test_8000_filepath)
        downsample_sound = kaldi.resample_waveform(sound, sample_rate, sample_rate // 2)
        self.assertTrue(downsample_sound.size(-1) == sound.size(-1) // 2)

    def test_resample_waveform_identity_size(self):
        sound, sample_rate = torchaudio.load_wav(self.test_8000_filepath)
        downsample_sound = kaldi.resample_waveform(sound, sample_rate, sample_rate)
        self.assertTrue(downsample_sound.size(-1) == sound.size(-1))

    def _test_resample_waveform_accuracy(self, up_scale_factor=None, down_scale_factor=None,
                                         atol=1e-1, rtol=1e-4):
        # resample the signal and compare it to the ground truth
        n_to_trim = 20
        sample_rate = 1000
        new_sample_rate = sample_rate

        if up_scale_factor is not None:
            new_sample_rate *= up_scale_factor

        if down_scale_factor is not None:
            new_sample_rate //= down_scale_factor

        duration = 5  # seconds
        original_timestamps = torch.arange(0, duration, 1.0 / sample_rate)

        sound = 123 * torch.cos(2 * math.pi * 3 * original_timestamps).unsqueeze(0)
        estimate = kaldi.resample_waveform(sound, sample_rate, new_sample_rate).squeeze()

        new_timestamps = torch.arange(0, duration, 1.0 / new_sample_rate)[:estimate.size(0)]
        ground_truth = 123 * torch.cos(2 * math.pi * 3 * new_timestamps)

        # trim the first/last n samples as these points have boundary effects
        ground_truth = ground_truth[..., n_to_trim:-n_to_trim]
        estimate = estimate[..., n_to_trim:-n_to_trim]

        torch.testing.assert_allclose(estimate, ground_truth, atol=atol, rtol=rtol)

    def test_resample_waveform_downsample_accuracy(self):
        for i in range(1, 20):
            self._test_resample_waveform_accuracy(down_scale_factor=i * 2)

    def test_resample_waveform_upsample_accuracy(self):
        for i in range(1, 20):
            self._test_resample_waveform_accuracy(up_scale_factor=1.0 + i / 20.0)

    def test_resample_waveform_multi_channel(self):
        num_channels = 3

        sound, sample_rate = torchaudio.load_wav(self.test_8000_filepath)  # (1, 8000)
        multi_sound = sound.repeat(num_channels, 1)  # (num_channels, 8000)

        for i in range(num_channels):
            multi_sound[i, :] *= (i + 1) * 1.5

        multi_sound_sampled = kaldi.resample_waveform(multi_sound, sample_rate, sample_rate // 2)

        # check that sampling is same whether using separately or in a tensor of size (c, n)
        for i in range(num_channels):
            single_channel = sound * (i + 1) * 1.5
            single_channel_sampled = kaldi.resample_waveform(single_channel, sample_rate, sample_rate // 2)
            torch.testing.assert_allclose(multi_sound_sampled[i, :], single_channel_sampled[0], rtol=1e-4, atol=1e-8)
