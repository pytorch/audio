import torch
import torchaudio.transforms as T
from parameterized import parameterized

from torchaudio_unittest.common_utils import (
    skipIfNoSoxBackend,
    skipIfNoExec,
    TempDirMixin,
    TorchaudioTestCase,
    get_asset_path,
    sox_utils,
    load_wav,
    get_whitenoise,
)


@skipIfNoSoxBackend
@skipIfNoExec('sox')
class TestFunctionalFiltering(TempDirMixin, TorchaudioTestCase):
    def run_sox_effect(self, input_file, effect):
        output_file = self.get_temp_path('expected.wav')
        sox_utils.run_sox_effect(input_file, output_file, [str(e) for e in effect])
        return load_wav(output_file)

    def assert_sox_effect(self, result, input_path, effects, atol=1e-04, rtol=1e-5):
        expected, _ = self.run_sox_effect(input_path, effects)
        self.assertEqual(result, expected, atol=atol, rtol=rtol)

    @parameterized.expand([
        ('q', 'quarter_sine'),
        ('h', 'half_sine'),
        ('t', 'linear'),
    ])
    def test_fade(self, fade_shape_sox, fade_shape):
        fade_in_len, fade_out_len = 44100, 44100
        data, path = self.get_whitenoise(sample_rate=44100)
        result = T.Fade(fade_in_len, fade_out_len, fade_shape)(data)
        self.assert_sox_effect(result, path, ['fade', fade_shape_sox, '1', '0', '1'])

    @parameterized.expand([
        ('amplitude', 1.1),
        ('db', 2),
        ('power', 2),
    ])
    def test_vol(self, gain_type, gain):
        data, path = self.get_whitenoise()
        result = T.Vol(gain, gain_type)(data)
        self.assert_sox_effect(result, path, ['vol', f'{gain}', gain_type])

    @parameterized.expand(['vad-go-stereo-44100.wav', 'vad-go-mono-32000.wav'])
    def test_vad(self, filename):
        path = get_asset_path(filename)
        data, sample_rate = load_wav(path)
        result = T.Vad(sample_rate)(data)
        self.assert_sox_effect(result, path, ['vad'])
