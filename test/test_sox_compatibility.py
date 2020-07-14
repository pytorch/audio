import unittest

import torch
import torchaudio.functional as F

from .common_utils import (
    skipIfNoSoxBackend,
    skipIfNoExec,
    TempDirMixin,
    TorchaudioTestCase,
    get_asset_path,
    sox_utils,
    load_wav,
    save_wav,
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

    def get_whitenoise(self, sample_rate=8000):
        noise = get_whitenoise(
            sample_rate=sample_rate, duration=3, scale_factor=0.9,
        )
        path = self.get_temp_path("whitenoise.wav")
        save_wav(path, noise, sample_rate)
        return noise, path

    def test_gain(self):
        path = get_asset_path('steam-train-whistle-daniel_simon.wav')
        data, _ = load_wav(path)
        result = F.gain(data, 3)
        self.assert_sox_effect(result, path, ['gain', 3])

    def test_dither(self):
        path = get_asset_path('steam-train-whistle-daniel_simon.wav')
        data, _ = load_wav(path)
        result = F.dither(data)
        self.assert_sox_effect(result, path, ['dither'])

    def test_dither_noise(self):
        path = get_asset_path('steam-train-whistle-daniel_simon.wav')
        data, _ = load_wav(path)
        result = F.dither(data, noise_shaping=True)
        self.assert_sox_effect(result, path, ['dither', '-s'], atol=1.5e-4)

    def test_lowpass(self):
        cutoff_freq = 3000
        sample_rate = 8000

        data, path = self.get_whitenoise(sample_rate)
        result = F.lowpass_biquad(data, sample_rate, cutoff_freq)
        self.assert_sox_effect(result, path, ['lowpass', cutoff_freq], atol=1.5e-4)

    def test_highpass(self):
        cutoff_freq = 2000
        sample_rate = 8000

        data, path = self.get_whitenoise(sample_rate)
        result = F.highpass_biquad(data, sample_rate, cutoff_freq)
        self.assert_sox_effect(result, path, ['highpass', cutoff_freq], atol=1.5e-4)

    def test_allpass(self):
        central_freq = 1000
        q = 0.707
        sample_rate = 8000

        data, path = self.get_whitenoise(sample_rate)
        result = F.allpass_biquad(data, sample_rate, central_freq, q)
        self.assert_sox_effect(result, path, ['allpass', central_freq, f'{q}q'])

    def test_bandpass_with_csg(self):
        central_freq = 1000
        q = 0.707
        const_skirt_gain = True
        sample_rate = 8000

        data, path = self.get_whitenoise(sample_rate)
        result = F.bandpass_biquad(data, sample_rate, central_freq, q, const_skirt_gain)
        self.assert_sox_effect(result, path, ['bandpass', '-c', central_freq, f'{q}q'])

    def test_bandpass_without_csg(self):
        central_freq = 1000
        q = 0.707
        const_skirt_gain = False
        sample_rate = 8000

        data, path = self.get_whitenoise(sample_rate)
        result = F.bandpass_biquad(data, sample_rate, central_freq, q, const_skirt_gain)
        self.assert_sox_effect(result, path, ['bandpass', central_freq, f'{q}q'])

    def test_bandreject(self):
        central_freq = 1000
        q = 0.707
        sample_rate = 8000

        data, path = self.get_whitenoise(sample_rate)
        result = F.bandreject_biquad(data, sample_rate, central_freq, q)
        self.assert_sox_effect(result, path, ['bandreject', central_freq, f'{q}q'])

    def test_band_with_noise(self):
        central_freq = 1000
        q = 0.707
        noise = True
        sample_rate = 8000

        data, path = self.get_whitenoise(sample_rate)
        result = F.band_biquad(data, sample_rate, central_freq, q, noise)
        self.assert_sox_effect(result, path, ['band', '-n', central_freq, f'{q}q'])

    def test_band_without_noise(self):
        central_freq = 1000
        q = 0.707
        noise = False
        sample_rate = 8000

        data, path = self.get_whitenoise(sample_rate)
        result = F.band_biquad(data, sample_rate, central_freq, q, noise)
        self.assert_sox_effect(result, path, ['band', central_freq, f'{q}q'])

    def test_treble(self):
        central_freq = 1000
        q = 0.707
        gain = 40
        sample_rate = 8000

        data, path = self.get_whitenoise(sample_rate)
        result = F.treble_biquad(data, sample_rate, gain, central_freq, q)
        self.assert_sox_effect(result, path, ['treble', gain, central_freq, f'{q}q'])

    def test_bass(self):
        central_freq = 1000
        q = 0.707
        gain = 40
        sample_rate = 8000

        data, path = self.get_whitenoise(sample_rate)
        result = F.bass_biquad(data, sample_rate, gain, central_freq, q)
        self.assert_sox_effect(result, path, ['bass', gain, central_freq, f'{q}q'], atol=1.5e-4)

    def test_deemph(self):
        sample_rate = 44100
        data, path = self.get_whitenoise(sample_rate)
        result = F.deemph_biquad(data, sample_rate)
        self.assert_sox_effect(result, path, ['deemph'])

    def test_riaa(self):
        sample_rate = 44100
        data, path = self.get_whitenoise(sample_rate)
        result = F.riaa_biquad(data, sample_rate)
        self.assert_sox_effect(result, path, ['riaa'])

    def test_contrast(self):
        enhancement_amount = 80.

        data, path = self.get_whitenoise()
        result = F.contrast(data, enhancement_amount)
        self.assert_sox_effect(result, path, ['contrast', enhancement_amount])

    def test_dcshift_with_limiter(self):
        shift = 0.5
        limiter_gain = 0.05

        data, path = self.get_whitenoise()
        result = F.dcshift(data, shift, limiter_gain)
        self.assert_sox_effect(result, path, ['dcshift', shift, limiter_gain])

    def test_dcshift_without_limiter(self):
        shift = 0.6

        data, path = self.get_whitenoise()
        result = F.dcshift(data, shift)
        self.assert_sox_effect(result, path, ['dcshift', shift])

    def test_overdrive(self):
        gain = 30
        colour = 40

        data, path = self.get_whitenoise()
        result = F.overdrive(data, gain, colour)
        self.assert_sox_effect(result, path, ['overdrive', gain, colour])

    def test_phaser_sine(self):
        gain_in = 0.5
        gain_out = 0.8
        delay_ms = 2.0
        decay = 0.4
        speed = 0.5
        sample_rate = 8000

        data, path = self.get_whitenoise(sample_rate)
        result = F.phaser(data, sample_rate, gain_in, gain_out, delay_ms, decay, speed, sinusoidal=True)
        self.assert_sox_effect(result, path, ['phaser', gain_in, gain_out, delay_ms, decay, speed, '-s'])

    def test_phaser_triangle(self):
        gain_in = 0.5
        gain_out = 0.8
        delay_ms = 2.0
        decay = 0.4
        speed = 0.5
        sample_rate = 8000

        data, path = self.get_whitenoise(sample_rate)
        result = F.phaser(data, sample_rate, gain_in, gain_out, delay_ms, decay, speed, sinusoidal=False)
        self.assert_sox_effect(result, path, ['phaser', gain_in, gain_out, delay_ms, decay, speed, '-t'])

    def test_flanger_triangle_linear(self):
        delay = 0.6
        depth = 0.87
        regen = 3.0
        width = 0.9
        speed = 0.5
        phase = 30
        sample_rate = 8000

        data, path = self.get_whitenoise(sample_rate)
        result = F.flanger(
            data, sample_rate, delay, depth, regen, width, speed, phase,
            modulation='triangular', interpolation='linear')
        self.assert_sox_effect(
            result, path, ['flanger', delay, depth, regen, width, speed, 'triangle', phase, 'linear'])

    def test_flanger_triangle_quad(self):
        delay = 0.8
        depth = 0.88
        regen = 3.0
        width = 0.4
        speed = 0.5
        phase = 40
        sample_rate = 8000

        data, path = self.get_whitenoise(sample_rate)
        result = F.flanger(
            data, sample_rate, delay, depth, regen, width, speed, phase,
            modulation='triangular', interpolation='quadratic')
        self.assert_sox_effect(
            result, path, ['flanger', delay, depth, regen, width, speed, 'triangle', phase, 'quadratic'])

    def test_flanger_sine_linear(self):
        delay = 0.8
        depth = 0.88
        regen = 3.0
        width = 0.23
        speed = 1.3
        phase = 60
        sample_rate = 8000

        data, path = self.get_whitenoise(sample_rate)
        result = F.flanger(
            data, sample_rate, delay, depth, regen, width, speed, phase,
            modulation='sinusoidal', interpolation='linear')
        self.assert_sox_effect(
            result, path, ['flanger', delay, depth, regen, width, speed, 'sine', phase, 'linear'])

    def test_flanger_sine_quad(self):
        delay = 0.9
        depth = 0.9
        regen = 4.0
        width = 0.23
        speed = 1.3
        phase = 25
        sample_rate = 8000

        data, path = self.get_whitenoise(sample_rate)
        result = F.flanger(
            data, sample_rate, delay, depth, regen, width, speed, phase,
            modulation='sinusoidal', interpolation='quadratic')
        self.assert_sox_effect(
            result, path, ['flanger', delay, depth, regen, width, speed, 'sine', phase, 'quadratic'])

    def test_equalizer(self):
        center_freq = 300
        q = 0.707
        gain = 1
        sample_rate = 8000

        data, path = self.get_whitenoise(sample_rate)
        result = F.equalizer_biquad(data, sample_rate, center_freq, gain, q)
        self.assert_sox_effect(result, path, ['equalizer', center_freq, q, gain])

    def test_perf_biquad_filtering(self):
        b0 = 0.4
        b1 = 0.2
        b2 = 0.9
        a0 = 0.7
        a1 = 0.2
        a2 = 0.6

        data, path = self.get_whitenoise()
        result = F.lfilter(data, torch.tensor([a0, a1, a2]), torch.tensor([b0, b1, b2]))
        self.assert_sox_effect(result, path, ['biquad', b0, b1, b2, a0, a1, a2])


if __name__ == "__main__":
    unittest.main()
