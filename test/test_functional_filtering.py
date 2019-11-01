from __future__ import absolute_import, division, print_function, unicode_literals
import math
import os
import torch
import torchaudio
import torchaudio.functional as F
import unittest
import common_utils
import time


class TestFunctionalFiltering(unittest.TestCase):
    test_dirpath, test_dir = common_utils.create_temp_assets_dir()

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

        assert torch.allclose(waveform[:, 0:-3], output_waveform[:, 3:], atol=1e-5)

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

        filepath = os.path.join(self.test_dirpath, "assets", "whitenoise.mp3")
        waveform, _ = torchaudio.load(filepath, normalization=True)

        self._test_lfilter(waveform, torch.device("cpu"))

    def test_lfilter_gpu(self):
        if torch.cuda.is_available():
            filepath = os.path.join(self.test_dirpath, "assets", "whitenoise.mp3")
            waveform, _ = torchaudio.load(filepath, normalization=True)
            cuda0 = torch.device("cuda:0")
            cuda_waveform = waveform.cuda(device=cuda0)
            self._test_lfilter(cuda_waveform, cuda0)
        else:
            print("skipping GPU test for lfilter because device not available")
            pass

    def test_lowpass(self):

        """
        Test biquad lowpass filter, compare to SoX implementation
        """

        CUTOFF_FREQ = 3000

        noise_filepath = os.path.join(self.test_dirpath, "assets", "whitenoise.mp3")
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("lowpass", [CUTOFF_FREQ])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.lowpass_biquad(waveform, sample_rate, CUTOFF_FREQ)

        assert torch.allclose(sox_output_waveform, output_waveform, atol=1e-4)

    def test_highpass(self):
        """
        Test biquad highpass filter, compare to SoX implementation
        """

        CUTOFF_FREQ = 2000

        noise_filepath = os.path.join(self.test_dirpath, "assets", "whitenoise.mp3")
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("highpass", [CUTOFF_FREQ])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.highpass_biquad(waveform, sample_rate, CUTOFF_FREQ)

        # TBD - this fails at the 1e-4 level, debug why
        assert torch.allclose(sox_output_waveform, output_waveform, atol=1e-3)

    def test_equalizer(self):
        """
        Test biquad peaking equalizer filter, compare to SoX implementation
        """

        CENTER_FREQ = 300
        Q = 0.707
        GAIN = 1

        noise_filepath = os.path.join(self.test_dirpath, "assets", "whitenoise.mp3")
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("equalizer", [CENTER_FREQ, Q, GAIN])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        output_waveform = F.equalizer_biquad(waveform, sample_rate, CENTER_FREQ, GAIN, Q)

        assert torch.allclose(sox_output_waveform, output_waveform, atol=1e-4)

    def test_perf_biquad_filtering(self):

        fn_sine = os.path.join(self.test_dirpath, "assets", "whitenoise.mp3")

        b0 = 0.4
        b1 = 0.2
        b2 = 0.9
        a0 = 0.7
        a1 = 0.2
        a2 = 0.6

        # SoX method
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(fn_sine)
        _timing_sox = time.time()
        E.append_effect_to_chain("biquad", [b0, b1, b2, a0, a1, a2])
        waveform_sox_out, sr = E.sox_build_flow_effects()
        _timing_sox_run_time = time.time() - _timing_sox

        _timing_lfilter_filtering = time.time()
        waveform, sample_rate = torchaudio.load(fn_sine, normalization=True)
        waveform_lfilter_out = F.lfilter(
            waveform, torch.tensor([a0, a1, a2]), torch.tensor([b0, b1, b2])
        )
        _timing_lfilter_run_time = time.time() - _timing_lfilter_filtering

        assert torch.allclose(waveform_sox_out, waveform_lfilter_out, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
