"""Test suites for checking numerical compatibility against Kaldi"""
import json
import shutil
import unittest
import subprocess
import math

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

    @parameterized.expand(_load_params(common_utils.get_asset_path('kaldi_test_spectrogram_args.json')))
    @unittest.skipIf(_not_available('compute-spectrogram-feats'), '`compute-spectrogram-feats` not available')
    def test_spectrogram(self, kwargs):
        """spectrogram should be numerically compatible with compute-spectrogram-feats"""
        wave_file = common_utils.get_asset_path('kaldi_file.wav')
        waveform = torchaudio.load_wav(wave_file)[0].to(dtype=self.dtype, device=self.device)
        result = torchaudio.compliance.kaldi.spectrogram(waveform, **kwargs)
        command = ['compute-spectrogram-feats'] + _convert_args(**kwargs) + ['scp:-', 'ark:-']
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

    def test_mfcc_empty(self):
        # Passing in an empty tensor should result in an error
        self.assertRaises(AssertionError, torchaudio.compliance.kaldi.mfcc, torch.empty(0))

    def test_resample_waveform_upsample_size(self):
        test_8000_filepath = common_utils.get_asset_path('kaldi_file_8000.wav')
        sound, sample_rate = torchaudio.load_wav(test_8000_filepath)
        upsample_sound = torchaudio.compliance.kaldi.resample_waveform(sound, sample_rate, sample_rate * 2)
        self.assertTrue(upsample_sound.size(-1) == sound.size(-1) * 2)

    def test_resample_waveform_downsample_size(self):
        test_8000_filepath = common_utils.get_asset_path('kaldi_file_8000.wav')
        sound, sample_rate = torchaudio.load_wav(test_8000_filepath)
        downsample_sound = torchaudio.compliance.kaldi.resample_waveform(sound, sample_rate, sample_rate // 2)
        self.assertTrue(downsample_sound.size(-1) == sound.size(-1) // 2)

    def test_resample_waveform_identity_size(self):
        test_8000_filepath = common_utils.get_asset_path('kaldi_file_8000.wav')
        sound, sample_rate = torchaudio.load_wav(test_8000_filepath)
        downsample_sound = torchaudio.compliance.kaldi.resample_waveform(sound, sample_rate, sample_rate)
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
        estimate = torchaudio.compliance.kaldi.resample_waveform(sound, sample_rate, new_sample_rate).squeeze()

        new_timestamps = torch.arange(0, duration, 1.0 / new_sample_rate)[:estimate.size(0)]
        ground_truth = 123 * torch.cos(2 * math.pi * 3 * new_timestamps)

        # trim the first/last n samples as these points have boundary effects
        ground_truth = ground_truth[..., n_to_trim:-n_to_trim]
        estimate = estimate[..., n_to_trim:-n_to_trim]

        self.assert_equal(estimate, expected=ground_truth, atol=atol, rtol=rtol)

    def test_resample_waveform_downsample_accuracy(self):
        for i in range(1, 20):
            self._test_resample_waveform_accuracy(down_scale_factor=i * 2)

    def test_resample_waveform_upsample_accuracy(self):
        for i in range(1, 20):
            self._test_resample_waveform_accuracy(up_scale_factor=1.0 + i / 20.0)

    def test_resample_waveform_multi_channel(self):
        num_channels = 3

        test_8000_filepath = common_utils.get_asset_path('kaldi_file_8000.wav')
        sound, sample_rate = torchaudio.load_wav(test_8000_filepath)  # (1, 8000)
        multi_sound = sound.repeat(num_channels, 1)  # (num_channels, 8000)

        for i in range(num_channels):
            multi_sound[i, :] *= (i + 1) * 1.5

        multi_sound_sampled = torchaudio.compliance.kaldi.resample_waveform(multi_sound, sample_rate, sample_rate // 2)

        # check that sampling is same whether using separately or in a tensor of size (c, n)
        for i in range(num_channels):
            single_channel = sound * (i + 1) * 1.5
            single_channel_sampled = torchaudio.compliance.kaldi.resample_waveform(single_channel,
                                                                                   sample_rate,
                                                                                   sample_rate // 2)
            self.assert_equal(multi_sound_sampled[i, :], expected=single_channel_sampled[0], rtol=1e-4, atol=1e-8)
