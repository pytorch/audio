"""Test suites for checking numerical compatibility against Kaldi"""
import subprocess

import kaldi_io
import torch
import torchaudio.functional as F
import torchaudio.compliance.kaldi
from parameterized import parameterized

from torchaudio_unittest.common_utils import (
    TestBaseMixin,
    TempDirMixin,
    load_params,
    skipIfNoExec,
    get_asset_path,
    load_wav,
    save_wav,
    get_sinusoid,
)


def _convert_args(**kwargs):
    args = []
    for key, value in kwargs.items():
        if key == 'sample_rate':
            key = 'sample_frequency'
        key = '--' + key.replace('_', '-')
        value = str(value).lower() if value in [True, False] else str(value)
        args.append('%s=%s' % (key, value))
    return args


def _run_kaldi(command, input_type, input_value):
    """Run provided Kaldi command, pass a tensor and get the resulting tensor

    Args:
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


class KaldiTestBase(TempDirMixin, TestBaseMixin):
    def assert_equal(self, output, *, expected, rtol=None, atol=None):
        expected = expected.to(dtype=self.dtype, device=self.device)
        self.assertEqual(output, expected, rtol=rtol, atol=atol)


class Kaldi(KaldiTestBase):
    @skipIfNoExec('apply-cmvn-sliding')
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

    @parameterized.expand(load_params('kaldi_test_fbank_args.json'))
    @skipIfNoExec('compute-fbank-feats')
    def test_fbank(self, kwargs):
        """fbank should be numerically compatible with compute-fbank-feats"""
        wave_file = get_asset_path('kaldi_file.wav')
        waveform = load_wav(wave_file, normalize=False)[0].to(dtype=self.dtype, device=self.device)
        result = torchaudio.compliance.kaldi.fbank(waveform, **kwargs)
        command = ['compute-fbank-feats'] + _convert_args(**kwargs) + ['scp:-', 'ark:-']
        kaldi_result = _run_kaldi(command, 'scp', wave_file)
        self.assert_equal(result, expected=kaldi_result, rtol=1e-4, atol=1e-8)

    @parameterized.expand(load_params('kaldi_test_spectrogram_args.json'))
    @skipIfNoExec('compute-spectrogram-feats')
    def test_spectrogram(self, kwargs):
        """spectrogram should be numerically compatible with compute-spectrogram-feats"""
        wave_file = get_asset_path('kaldi_file.wav')
        waveform = load_wav(wave_file, normalize=False)[0].to(dtype=self.dtype, device=self.device)
        result = torchaudio.compliance.kaldi.spectrogram(waveform, **kwargs)
        command = ['compute-spectrogram-feats'] + _convert_args(**kwargs) + ['scp:-', 'ark:-']
        kaldi_result = _run_kaldi(command, 'scp', wave_file)
        self.assert_equal(result, expected=kaldi_result, rtol=1e-4, atol=1e-8)

    @parameterized.expand(load_params('kaldi_test_mfcc_args.json'))
    @skipIfNoExec('compute-mfcc-feats')
    def test_mfcc(self, kwargs):
        """mfcc should be numerically compatible with compute-mfcc-feats"""
        wave_file = get_asset_path('kaldi_file.wav')
        waveform = load_wav(wave_file, normalize=False)[0].to(dtype=self.dtype, device=self.device)
        result = torchaudio.compliance.kaldi.mfcc(waveform, **kwargs)
        command = ['compute-mfcc-feats'] + _convert_args(**kwargs) + ['scp:-', 'ark:-']
        kaldi_result = _run_kaldi(command, 'scp', wave_file)
        self.assert_equal(result, expected=kaldi_result, rtol=1e-4, atol=1e-8)


class KaldiCPUOnly(KaldiTestBase):
    @parameterized.expand(load_params('kaldi_test_pitch_args.json'))
    @skipIfNoExec('compute-kaldi-pitch-feats')
    def test_pitch_feats(self, kwargs):
        """compute_kaldi_pitch produces numerically compatible result with compute-kaldi-pitch-feats"""
        sample_rate = kwargs['sample_rate']
        waveform = get_sinusoid(dtype='float32', sample_rate=sample_rate)
        result = F.compute_kaldi_pitch(waveform[0], **kwargs)

        waveform = get_sinusoid(dtype='int16', sample_rate=sample_rate)
        wave_file = self.get_temp_path('test.wav')
        save_wav(wave_file, waveform, sample_rate)

        command = ['compute-kaldi-pitch-feats'] + _convert_args(**kwargs) + ['scp:-', 'ark:-']
        kaldi_result = _run_kaldi(command, 'scp', wave_file)
        self.assert_equal(result, expected=kaldi_result)
