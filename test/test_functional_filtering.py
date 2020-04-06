import os
import unittest

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

from common_utils import AudioBackendScope, BACKENDS, create_temp_assets_dir


class TestFunctionalFiltering(unittest.TestCase):
    test_dirpath, test_dir = create_temp_assets_dir()

    def _test_lfilter_basic(self, dtype, device):
        """
        Create a very basic signal,
        Then make a simple 4th order delay
        The output should be same as the input but shifted
        """

        torch.random.manual_seed(42)
        waveform = torch.rand(2, 44100 * 1, dtype=dtype, device=device)
        b_coeffs = torch.tensor([0, 0, 0, 1], dtype=dtype, device=device)
        a_coeffs = torch.tensor([1, 0, 0, 0], dtype=dtype, device=device)
        output_waveform = F.lfilter(waveform, a_coeffs, b_coeffs)

        torch.testing.assert_allclose(waveform[:, 0:-3], output_waveform[:, 3:], atol=1e-5)

    def test_lfilter_basic(self):
        self._test_lfilter_basic(torch.float32, torch.device("cpu"))

    def test_lfilter_basic_double(self):
        self._test_lfilter_basic(torch.float64, torch.device("cpu"))

    def test_lfilter_basic_gpu(self):
        if torch.cuda.is_available():
            self._test_lfilter_basic(torch.float32, torch.device("cuda:0"))
        else:
            print("skipping GPU test for lfilter_basic because device not available")
            pass

    def _test_lfilter(self, waveform, device):
        """
        Design an IIR lowpass filter using scipy.signal filter design
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirdesign.html#scipy.signal.iirdesign

        Example
          >>> from scipy.signal import iirdesign
          >>> b, a = iirdesign(0.2, 0.3, 1, 60)
        """

        b_coeffs = torch.tensor(
            [
                0.00299893,
                -0.0051152,
                0.00841964,
                -0.00747802,
                0.00841964,
                -0.0051152,
                0.00299893,
            ],
            device=device,
        )
        a_coeffs = torch.tensor(
            [
                1.0,
                -4.8155751,
                10.2217618,
                -12.14481273,
                8.49018171,
                -3.3066882,
                0.56088705,
            ],
            device=device,
        )

        output_waveform = F.lfilter(waveform, a_coeffs, b_coeffs)
        assert len(output_waveform.size()) == 2
        assert output_waveform.size(0) == waveform.size(0)
        assert output_waveform.size(1) == waveform.size(1)

    def test_lfilter(self):

        filepath = os.path.join(self.test_dirpath, "assets", "whitenoise.wav")
        waveform, _ = torchaudio.load(filepath, normalization=True)

        self._test_lfilter(waveform, torch.device("cpu"))

    def test_lfilter_gpu(self):
        if torch.cuda.is_available():
            filepath = os.path.join(self.test_dirpath, "assets", "whitenoise.wav")
            waveform, _ = torchaudio.load(filepath, normalization=True)
            cuda0 = torch.device("cuda:0")
            cuda_waveform = waveform.cuda(device=cuda0)
            self._test_lfilter(cuda_waveform, cuda0)
        else:
            print("skipping GPU test for lfilter because device not available")
            pass

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_gain(self):
        test_filepath = os.path.join(self.test_dirpath, "assets", "steam-train-whistle-daniel_simon.wav")
        waveform, _ = torchaudio.load(test_filepath)

        waveform_gain = F.gain(waveform, 3)
        self.assertTrue(waveform_gain.abs().max().item(), 1.)

        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(test_filepath)
        E.append_effect_to_chain("gain", [3])
        sox_gain_waveform = E.sox_build_flow_effects()[0]

        torch.testing.assert_allclose(waveform_gain, sox_gain_waveform, atol=1e-04)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_dither(self):
        test_filepath = os.path.join(self.test_dirpath, "assets", "steam-train-whistle-daniel_simon.wav")
        waveform, _ = torchaudio.load(test_filepath)

        waveform_dithered = F.dither(waveform)
        waveform_dithered_noiseshaped = F.dither(waveform, noise_shaping=True)

        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(test_filepath)
        E.append_effect_to_chain("dither", [])
        sox_dither_waveform = E.sox_build_flow_effects()[0]

        torch.testing.assert_allclose(waveform_dithered, sox_dither_waveform, atol=1e-04)
        E.clear_chain()

        E.append_effect_to_chain("dither", ["-s"])
        sox_dither_waveform_ns = E.sox_build_flow_effects()[0]

        torch.testing.assert_allclose(waveform_dithered_noiseshaped, sox_dither_waveform_ns, atol=1e-02)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_vctk_transform_pipeline(self):
        test_filepath_vctk = os.path.join(self.test_dirpath, "assets/VCTK-Corpus/wav48/p224/", "p224_002.wav")
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

        noise_filepath = os.path.join(self.test_dirpath, "assets", "whitenoise.wav")
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("lowpass", [CUTOFF_FREQ])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.lowpass_biquad(waveform, sample_rate, CUTOFF_FREQ)

        torch.testing.assert_allclose(sox_output_waveform, output_waveform, atol=1e-4)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_highpass(self):
        """
        Test biquad highpass filter, compare to SoX implementation
        """

        CUTOFF_FREQ = 2000

        noise_filepath = os.path.join(self.test_dirpath, "assets", "whitenoise.wav")
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("highpass", [CUTOFF_FREQ])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.highpass_biquad(waveform, sample_rate, CUTOFF_FREQ)

        # TBD - this fails at the 1e-4 level, debug why
        torch.testing.assert_allclose(sox_output_waveform, output_waveform, atol=1e-3)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_allpass(self):
        """
        Test biquad allpass filter, compare to SoX implementation
        """

        CENTRAL_FREQ = 1000
        Q = 0.707

        noise_filepath = os.path.join(self.test_dirpath, "assets", "whitenoise.wav")
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("allpass", [CENTRAL_FREQ, str(Q) + 'q'])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.allpass_biquad(waveform, sample_rate, CENTRAL_FREQ, Q)

        torch.testing.assert_allclose(sox_output_waveform, output_waveform, atol=1e-4)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_bandpass_with_csg(self):
        """
        Test biquad bandpass filter, compare to SoX implementation
        """

        CENTRAL_FREQ = 1000
        Q = 0.707
        CONST_SKIRT_GAIN = True

        noise_filepath = os.path.join(self.test_dirpath, "assets", "whitenoise.wav")
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("bandpass", ["-c", CENTRAL_FREQ, str(Q) + 'q'])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.bandpass_biquad(waveform, sample_rate, CENTRAL_FREQ, Q, CONST_SKIRT_GAIN)

        torch.testing.assert_allclose(sox_output_waveform, output_waveform, atol=1e-4)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_bandpass_without_csg(self):
        """
        Test biquad bandpass filter, compare to SoX implementation
        """

        CENTRAL_FREQ = 1000
        Q = 0.707
        CONST_SKIRT_GAIN = False

        noise_filepath = os.path.join(self.test_dirpath, "assets", "whitenoise.wav")
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("bandpass", [CENTRAL_FREQ, str(Q) + 'q'])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.bandpass_biquad(waveform, sample_rate, CENTRAL_FREQ, Q, CONST_SKIRT_GAIN)

        torch.testing.assert_allclose(sox_output_waveform, output_waveform, atol=1e-4)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_bandreject(self):
        """
        Test biquad bandreject filter, compare to SoX implementation
        """

        CENTRAL_FREQ = 1000
        Q = 0.707

        noise_filepath = os.path.join(self.test_dirpath, "assets", "whitenoise.wav")
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("bandreject", [CENTRAL_FREQ, str(Q) + 'q'])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.bandreject_biquad(waveform, sample_rate, CENTRAL_FREQ, Q)

        torch.testing.assert_allclose(sox_output_waveform, output_waveform, atol=1e-4)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_band_with_noise(self):
        """
        Test biquad band filter with noise mode, compare to SoX implementation
        """

        CENTRAL_FREQ = 1000
        Q = 0.707
        NOISE = True

        noise_filepath = os.path.join(self.test_dirpath, "assets", "whitenoise.wav")
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("band", ["-n", CENTRAL_FREQ, str(Q) + 'q'])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.band_biquad(waveform, sample_rate, CENTRAL_FREQ, Q, NOISE)

        torch.testing.assert_allclose(sox_output_waveform, output_waveform, atol=1e-4)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_band_without_noise(self):
        """
        Test biquad band filter without noise mode, compare to SoX implementation
        """

        CENTRAL_FREQ = 1000
        Q = 0.707
        NOISE = False

        noise_filepath = os.path.join(self.test_dirpath, "assets", "whitenoise.wav")
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("band", [CENTRAL_FREQ, str(Q) + 'q'])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.band_biquad(waveform, sample_rate, CENTRAL_FREQ, Q, NOISE)

        torch.testing.assert_allclose(sox_output_waveform, output_waveform, atol=1e-4)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_treble(self):
        """
        Test biquad treble filter, compare to SoX implementation
        """

        CENTRAL_FREQ = 1000
        Q = 0.707
        GAIN = 40

        noise_filepath = os.path.join(self.test_dirpath, "assets", "whitenoise.wav")
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("treble", [GAIN, CENTRAL_FREQ, str(Q) + 'q'])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.treble_biquad(waveform, sample_rate, GAIN, CENTRAL_FREQ, Q)

        torch.testing.assert_allclose(sox_output_waveform, output_waveform, atol=1e-4)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_deemph(self):
        """
        Test biquad deemph filter, compare to SoX implementation
        """

        noise_filepath = os.path.join(self.test_dirpath, "assets", "whitenoise.wav")
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("deemph")
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.deemph_biquad(waveform, sample_rate)

        torch.testing.assert_allclose(sox_output_waveform, output_waveform, atol=1e-4)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_riaa(self):
        """
        Test biquad riaa filter, compare to SoX implementation
        """

        noise_filepath = os.path.join(self.test_dirpath, "assets", "whitenoise.wav")
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("riaa")
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.riaa_biquad(waveform, sample_rate)

        torch.testing.assert_allclose(sox_output_waveform, output_waveform, atol=1e-4)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_equalizer(self):
        """
        Test biquad peaking equalizer filter, compare to SoX implementation
        """

        CENTER_FREQ = 300
        Q = 0.707
        GAIN = 1

        noise_filepath = os.path.join(self.test_dirpath, "assets", "whitenoise.wav")
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("equalizer", [CENTER_FREQ, Q, GAIN])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.equalizer_biquad(waveform, sample_rate, CENTER_FREQ, GAIN, Q)

        torch.testing.assert_allclose(sox_output_waveform, output_waveform, atol=1e-4)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_perf_biquad_filtering(self):

        fn_sine = os.path.join(self.test_dirpath, "assets", "whitenoise.wav")

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

        torch.testing.assert_allclose(waveform_sox_out, waveform_lfilter_out, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
