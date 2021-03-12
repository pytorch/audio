from parameterized import parameterized
import torchaudio.functional as F

from torchaudio_unittest.common_utils import (
    get_sinusoid,
    load_params,
    save_wav,
    skipIfNoExec,
    TempDirMixin,
    TestBaseMixin,
)
from torchaudio_unittest.common_utils.kaldi_utils import (
    convert_args,
    run_kaldi,
)


class KaldiCPUOnly(TempDirMixin, TestBaseMixin):
    def assert_equal(self, output, *, expected, rtol=None, atol=None):
        expected = expected.to(dtype=self.dtype, device=self.device)
        self.assertEqual(output, expected, rtol=rtol, atol=atol)

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

        command = ['compute-kaldi-pitch-feats'] + convert_args(**kwargs) + ['scp:-', 'ark:-']
        kaldi_result = run_kaldi(command, 'scp', wave_file)
        self.assert_equal(result, expected=kaldi_result)
