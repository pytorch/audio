import warnings

import torch
import torchaudio.transforms as T
from parameterized import parameterized
from torchaudio_unittest.common_utils import (
    get_asset_path,
    get_whitenoise,
    load_wav,
    save_wav,
    skipIfNoExec,
    skipIfNoSox,
    sox_utils,
    TempDirMixin,
    TorchaudioTestCase,
)


@skipIfNoSox
@skipIfNoExec("sox")
class TestFunctionalFiltering(TempDirMixin, TorchaudioTestCase):
    def run_sox_effect(self, input_file, effect):
        output_file = self.get_temp_path("expected.wav")
        sox_utils.run_sox_effect(input_file, output_file, [str(e) for e in effect])
        return load_wav(output_file)

    def assert_sox_effect(self, result, input_path, effects, atol=1e-04, rtol=1e-5):
        expected, _ = self.run_sox_effect(input_path, effects)
        self.assertEqual(result, expected, atol=atol, rtol=rtol)

    def get_whitenoise(self, sample_rate=8000):
        noise = get_whitenoise(
            sample_rate=sample_rate,
            duration=3,
            scale_factor=0.9,
        )
        path = self.get_temp_path("whitenoise.wav")
        save_wav(path, noise, sample_rate)
        return noise, path

    @parameterized.expand(
        [
            ("q", "quarter_sine"),
            ("h", "half_sine"),
            ("t", "linear"),
        ]
    )
    def test_fade(self, fade_shape_sox, fade_shape):
        fade_in_len, fade_out_len = 44100, 44100
        data, path = self.get_whitenoise(sample_rate=44100)
        result = T.Fade(fade_in_len, fade_out_len, fade_shape)(data)
        self.assert_sox_effect(result, path, ["fade", fade_shape_sox, "1", "0", "1"])

    @parameterized.expand(
        [
            ("amplitude", 1.1),
            ("db", 2),
            ("power", 2),
        ]
    )
    def test_vol(self, gain_type, gain):
        data, path = self.get_whitenoise()
        result = T.Vol(gain, gain_type)(data)
        self.assert_sox_effect(result, path, ["vol", f"{gain}", gain_type])

    @parameterized.expand(["vad-go-stereo-44100.wav", "vad-go-mono-32000.wav"])
    def test_vad(self, filename):
        path = get_asset_path(filename)
        data, sample_rate = load_wav(path)
        result = T.Vad(sample_rate)(data)
        self.assert_sox_effect(result, path, ["vad"])

    @parameterized.expand(
        [
            (torch.zeros(32000), torch.zeros(0), 16000),
            (torch.zeros(1, 32000), torch.zeros(1, 0), 32000),
            (torch.zeros(2, 44100), torch.zeros(2, 0), 32000),
            (torch.zeros(2, 2, 44100), torch.zeros(2, 2, 0), 32000),
        ]
    )
    def test_vad_on_zero_audio(self, inpt: torch.Tensor, expected_output: torch.Tensor, sample_rate: int):
        result = T.Vad(sample_rate)(inpt)
        self.assertEqual(result, expected_output)

    def test_vad_warning(self):
        """vad should throw a warning if input dimension is greater than 2"""
        sample_rate = 41100

        data = torch.rand(5, 5, sample_rate)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            T.Vad(sample_rate)(data)
        assert len(w) == 1

        data = torch.rand(5, sample_rate)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            T.Vad(sample_rate)(data)
        assert len(w) == 0

        data = torch.rand(sample_rate)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            T.Vad(sample_rate)(data)
        assert len(w) == 0
