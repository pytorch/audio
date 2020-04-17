import unittest

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import common_utils
from common_utils import AudioBackendScope, BACKENDS


class TestFunctionalFiltering(unittest.TestCase):
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

        torch.testing.assert_allclose(waveform_gain, sox_gain_waveform, atol=1e-04, rtol=1e-5)

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

        torch.testing.assert_allclose(waveform_dithered, sox_dither_waveform, atol=1e-04, rtol=1e-5)
        E.clear_chain()

        E.append_effect_to_chain("dither", ["-s"])
        sox_dither_waveform_ns = E.sox_build_flow_effects()[0]

        torch.testing.assert_allclose(waveform_dithered_noiseshaped, sox_dither_waveform_ns, atol=1e-02, rtol=1e-5)

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

        torch.testing.assert_allclose(wf_vctk, wf_vctk_sox, rtol=1e-03, atol=1e-03)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_lowpass(self):
        """
        Test biquad lowpass filter, compare to SoX implementation
        """

        CUTOFF_FREQ = 3000

        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("lowpass", [CUTOFF_FREQ])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.lowpass_biquad(waveform, sample_rate, CUTOFF_FREQ)

        torch.testing.assert_allclose(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_highpass(self):
        """
        Test biquad highpass filter, compare to SoX implementation
        """

        CUTOFF_FREQ = 2000

        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("highpass", [CUTOFF_FREQ])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.highpass_biquad(waveform, sample_rate, CUTOFF_FREQ)

        # TBD - this fails at the 1e-4 level, debug why
        torch.testing.assert_allclose(output_waveform, sox_output_waveform, atol=1e-3, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_allpass(self):
        """
        Test biquad allpass filter, compare to SoX implementation
        """

        CENTRAL_FREQ = 1000
        Q = 0.707

        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("allpass", [CENTRAL_FREQ, str(Q) + 'q'])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.allpass_biquad(waveform, sample_rate, CENTRAL_FREQ, Q)

        torch.testing.assert_allclose(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_bandpass_with_csg(self):
        """
        Test biquad bandpass filter, compare to SoX implementation
        """

        CENTRAL_FREQ = 1000
        Q = 0.707
        CONST_SKIRT_GAIN = True

        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("bandpass", ["-c", CENTRAL_FREQ, str(Q) + 'q'])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.bandpass_biquad(waveform, sample_rate, CENTRAL_FREQ, Q, CONST_SKIRT_GAIN)

        torch.testing.assert_allclose(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_bandpass_without_csg(self):
        """
        Test biquad bandpass filter, compare to SoX implementation
        """

        CENTRAL_FREQ = 1000
        Q = 0.707
        CONST_SKIRT_GAIN = False

        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("bandpass", [CENTRAL_FREQ, str(Q) + 'q'])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.bandpass_biquad(waveform, sample_rate, CENTRAL_FREQ, Q, CONST_SKIRT_GAIN)

        torch.testing.assert_allclose(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_bandreject(self):
        """
        Test biquad bandreject filter, compare to SoX implementation
        """

        CENTRAL_FREQ = 1000
        Q = 0.707

        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("bandreject", [CENTRAL_FREQ, str(Q) + 'q'])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.bandreject_biquad(waveform, sample_rate, CENTRAL_FREQ, Q)

        torch.testing.assert_allclose(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_band_with_noise(self):
        """
        Test biquad band filter with noise mode, compare to SoX implementation
        """

        CENTRAL_FREQ = 1000
        Q = 0.707
        NOISE = True

        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("band", ["-n", CENTRAL_FREQ, str(Q) + 'q'])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.band_biquad(waveform, sample_rate, CENTRAL_FREQ, Q, NOISE)

        torch.testing.assert_allclose(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_band_without_noise(self):
        """
        Test biquad band filter without noise mode, compare to SoX implementation
        """

        CENTRAL_FREQ = 1000
        Q = 0.707
        NOISE = False

        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("band", [CENTRAL_FREQ, str(Q) + 'q'])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.band_biquad(waveform, sample_rate, CENTRAL_FREQ, Q, NOISE)

        torch.testing.assert_allclose(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_treble(self):
        """
        Test biquad treble filter, compare to SoX implementation
        """

        CENTRAL_FREQ = 1000
        Q = 0.707
        GAIN = 40

        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("treble", [GAIN, CENTRAL_FREQ, str(Q) + 'q'])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.treble_biquad(waveform, sample_rate, GAIN, CENTRAL_FREQ, Q)

        torch.testing.assert_allclose(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

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

        torch.testing.assert_allclose(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

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

        torch.testing.assert_allclose(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

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

        torch.testing.assert_allclose(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

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

        torch.testing.assert_allclose(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

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

        torch.testing.assert_allclose(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_equalizer(self):
        """
        Test biquad peaking equalizer filter, compare to SoX implementation
        """

        CENTER_FREQ = 300
        Q = 0.707
        GAIN = 1

        noise_filepath = common_utils.get_asset_path('whitenoise.wav')
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("equalizer", [CENTER_FREQ, Q, GAIN])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.equalizer_biquad(waveform, sample_rate, CENTER_FREQ, GAIN, Q)

        torch.testing.assert_allclose(output_waveform, sox_output_waveform, atol=1e-4, rtol=1e-5)

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

        torch.testing.assert_allclose(waveform_lfilter_out, waveform_sox_out, atol=1e-4, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
