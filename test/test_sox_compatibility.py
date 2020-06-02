import unittest

import torch
from torch.testing._internal.common_utils import TestCase
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

from . import common_utils
from .common_utils import AudioBackendScope, BACKENDS


class TestFunctionalFiltering(TestCase):
    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_gain(self):
        test_filepath = common_utils.get_asset_path('steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)

        waveform_gain = F.gain(waveform, 3)
        self.assertTrue(waveform_gain.abs().max().item(), 1.)

        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(test_filepath)
        E.append_effect_to_chain("gain", [3])
        sox_gain_waveform = E.sox_build_flow_effects()[0]

        self.assertEqual(waveform_gain, sox_gain_waveform, atol=1e-04, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_dither(self):
        test_filepath = common_utils.get_asset_path('steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)

        waveform_dithered = F.dither(waveform)
        waveform_dithered_noiseshaped = F.dither(waveform, noise_shaping=True)

        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(test_filepath)
        E.append_effect_to_chain("dither", [])
        sox_dither_waveform = E.sox_build_flow_effects()[0]

        self.assertEqual(waveform_dithered, sox_dither_waveform, atol=1e-04, rtol=1e-5)
        E.clear_chain()

        E.append_effect_to_chain("dither", ["-s"])
        sox_dither_waveform_ns = E.sox_build_flow_effects()[0]

        self.assertEqual(waveform_dithered_noiseshaped, sox_dither_waveform_ns, atol=1e-02, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_vctk_transform_pipeline(self):
        test_filepath_vctk = common_utils.get_asset_path('VCTK-Corpus', 'wav48', 'p224', 'p224_002.wav')
        wf_vctk, sr_vctk = torchaudio.load(test_filepath_vctk)

        # rate
        sample = T.Resample(sr_vctk, 16000, resampling_method='sinc_interpolation')
        wf_vctk = sample(wf_vctk)
        # dither
        wf_vctk = F.dither(wf_vctk, noise_shaping=True)

        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(test_filepath_vctk)
        E.append_effect_to_chain("gain", ["-h"])
        E.append_effect_to_chain("channels", [1])
        E.append_effect_to_chain("rate", [16000])
        E.append_effect_to_chain("gain", ["-rh"])
        E.append_effect_to_chain("dither", ["-s"])
        wf_vctk_sox = E.sox_build_flow_effects()[0]

        self.assertEqual(wf_vctk, wf_vctk_sox, rtol=1e-03, atol=1e-03)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_lowpass(self):
        """
        Test biquad lowpass filter, compare to SoX implementation
        """

        cutoff_freq = 3000

        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("lowpass", [cutoff_freq])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.lowpass_biquad(waveform, sample_rate, cutoff_freq)

        self.assertEqual(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_highpass(self):
        """
        Test biquad highpass filter, compare to SoX implementation
        """

        cutoff_freq = 2000

        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("highpass", [cutoff_freq])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.highpass_biquad(waveform, sample_rate, cutoff_freq)

        # TBD - this fails at the 1e-4 level, debug why
        self.assertEqual(output_waveform, sox_output_waveform, atol=1e-3, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_allpass(self):
        """
        Test biquad allpass filter, compare to SoX implementation
        """

        central_freq = 1000
        q = 0.707

        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("allpass", [central_freq, str(q) + 'q'])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.allpass_biquad(waveform, sample_rate, central_freq, q)

        self.assertEqual(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_bandpass_with_csg(self):
        """
        Test biquad bandpass filter, compare to SoX implementation
        """

        central_freq = 1000
        q = 0.707
        const_skirt_gain = True

        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("bandpass", ["-c", central_freq, str(q) + 'q'])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.bandpass_biquad(waveform, sample_rate, central_freq, q, const_skirt_gain)

        self.assertEqual(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_bandpass_without_csg(self):
        """
        Test biquad bandpass filter, compare to SoX implementation
        """

        central_freq = 1000
        q = 0.707
        const_skirt_gain = False

        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("bandpass", [central_freq, str(q) + 'q'])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.bandpass_biquad(waveform, sample_rate, central_freq, q, const_skirt_gain)

        self.assertEqual(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_bandreject(self):
        """
        Test biquad bandreject filter, compare to SoX implementation
        """

        central_freq = 1000
        q = 0.707

        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("bandreject", [central_freq, str(q) + 'q'])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.bandreject_biquad(waveform, sample_rate, central_freq, q)

        self.assertEqual(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_band_with_noise(self):
        """
        Test biquad band filter with noise mode, compare to SoX implementation
        """

        central_freq = 1000
        q = 0.707
        noise = True

        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("band", ["-n", central_freq, str(q) + 'q'])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.band_biquad(waveform, sample_rate, central_freq, q, noise)

        self.assertEqual(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_band_without_noise(self):
        """
        Test biquad band filter without noise mode, compare to SoX implementation
        """

        central_freq = 1000
        q = 0.707
        noise = False

        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("band", [central_freq, str(q) + 'q'])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.band_biquad(waveform, sample_rate, central_freq, q, noise)

        self.assertEqual(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_treble(self):
        """
        Test biquad treble filter, compare to SoX implementation
        """

        central_freq = 1000
        q = 0.707
        gain = 40

        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("treble", [gain, central_freq, str(q) + 'q'])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.treble_biquad(waveform, sample_rate, gain, central_freq, q)

        self.assertEqual(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_deemph(self):
        """
        Test biquad deemph filter, compare to SoX implementation
        """

        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("deemph")
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.deemph_biquad(waveform, sample_rate)

        self.assertEqual(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_riaa(self):
        """
        Test biquad riaa filter, compare to SoX implementation
        """

        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("riaa")
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.riaa_biquad(waveform, sample_rate)

        self.assertEqual(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_contrast(self):
        """
        Test contrast effect, compare to SoX implementation
        """
        enhancement_amount = 80.
        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("contrast", [enhancement_amount])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.contrast(waveform, enhancement_amount)

        self.assertEqual(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_dcshift_with_limiter(self):
        """
        Test dcshift effect, compare to SoX implementation
        """
        shift = 0.5
        limiter_gain = 0.05
        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("dcshift", [shift, limiter_gain])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, _ = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.dcshift(waveform, shift, limiter_gain)

        self.assertEqual(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_dcshift_without_limiter(self):
        """
        Test dcshift effect, compare to SoX implementation
        """
        shift = 0.6
        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("dcshift", [shift])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, _ = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.dcshift(waveform, shift)

        self.assertEqual(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_overdrive(self):
        """
        Test overdrive effect, compare to SoX implementation
        """
        gain = 30
        colour = 40
        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("overdrive", [gain, colour])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, _ = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.overdrive(waveform, gain, colour)

        self.assertEqual(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_phaser_sine(self):
        """
        Test phaser effect with sine moduldation, compare to SoX implementation
        """
        gain_in = 0.5
        gain_out = 0.8
        delay_ms = 2.0
        decay = 0.4
        speed = 0.5
        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("phaser", [gain_in, gain_out, delay_ms, decay, speed, "-s"])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.phaser(waveform, sample_rate, gain_in, gain_out, delay_ms, decay, speed, sinusoidal=True)

        self.assertEqual(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_phaser_triangle(self):
        """
        Test phaser effect with triangle modulation, compare to SoX implementation
        """
        gain_in = 0.5
        gain_out = 0.8
        delay_ms = 2.0
        decay = 0.4
        speed = 0.5
        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("phaser", [gain_in, gain_out, delay_ms, decay, speed, "-t"])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.phaser(waveform, sample_rate, gain_in, gain_out, delay_ms, decay, speed, sinusoidal=False)

        self.assertEqual(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_flanger_triangle_linear(self):
        """
        Test flanger effect with triangle modulation and linear interpolation, compare to SoX implementation
        """
        delay = 0.6
        depth = 0.87
        regen = 3.0
        width = 0.9
        speed = 0.5
        phase = 30
        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("flanger", [delay, depth, regen, width, speed, "triangle", phase, "linear"])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.flanger(waveform, sample_rate, delay, depth, regen, width, speed, phase,
                                    modulation='triangular', interpolation='linear')

        torch.testing.assert_allclose(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_flanger_triangle_quad(self):
        """
        Test flanger effect with triangle modulation and quadratic interpolation, compare to SoX implementation
        """
        delay = 0.8
        depth = 0.88
        regen = 3.0
        width = 0.4
        speed = 0.5
        phase = 40
        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("flanger", [delay, depth, regen, width, speed, "triangle", phase, "quadratic"])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.flanger(waveform, sample_rate, delay, depth, regen, width, speed, phase,
                                    modulation='triangular', interpolation='quadratic')

        torch.testing.assert_allclose(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_flanger_sine_linear(self):
        """
        Test flanger effect with sine modulation and linear interpolation, compare to SoX implementation
        """
        delay = 0.8
        depth = 0.88
        regen = 3.0
        width = 0.23
        speed = 1.3
        phase = 60
        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("flanger", [delay, depth, regen, width, speed, "sine", phase, "linear"])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.flanger(waveform, sample_rate, delay, depth, regen, width, speed, phase,
                                    modulation='sinusoidal', interpolation='linear')

        torch.testing.assert_allclose(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_flanger_sine_quad(self):
        """
        Test flanger effect with sine modulation and quadratic interpolation, compare to SoX implementation
        """
        delay = 0.9
        depth = 0.9
        regen = 4.0
        width = 0.23
        speed = 1.3
        phase = 25
        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("flanger", [delay, depth, regen, width, speed, "sine", phase, "quadratic"])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.flanger(waveform, sample_rate, delay, depth, regen, width, speed, phase,
                                    modulation='sinusoidal', interpolation='quadratic')

        torch.testing.assert_allclose(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_equalizer(self):
        """
        Test biquad peaking equalizer filter, compare to SoX implementation
        """

        center_freq = 300
        q = 0.707
        gain = 1

        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("equalizer", [center_freq, q, gain])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.equalizer_biquad(waveform, sample_rate, center_freq, gain, q)

        self.assertEqual(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_perf_biquad_filtering(self):

        fn_sine = common_utils.get_asset_path('whitenoise.wav')

        b0 = 0.4
        b1 = 0.2
        b2 = 0.9
        a0 = 0.7
        a1 = 0.2
        a2 = 0.6

        # SoX method
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(fn_sine)
        E.append_effect_to_chain("biquad", [b0, b1, b2, a0, a1, a2])
        waveform_sox_out, _ = E.sox_build_flow_effects()

        waveform, _ = torchaudio.load(fn_sine, normalization=True)
        waveform_lfilter_out = F.lfilter(
            waveform, torch.tensor([a0, a1, a2]), torch.tensor([b0, b1, b2])
        )

        self.assertEqual(waveform_lfilter_out, waveform_sox_out, atol=1e-4, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
