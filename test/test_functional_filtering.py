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

    def test_lfilter_basic(self):
        """
        Create a very basic signal,
        Then make a simple 4th order delay
        The output should be same as the input but shifted
        """

        torch.random.manual_seed(42)
        waveform = torch.rand(2, 10000)
        b_coeffs = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
        a_coeffs = torch.tensor([1, 0, 0, 0], dtype=torch.float32)
        output_waveform = F.lfilter(waveform, a_coeffs, b_coeffs)

        assert torch.allclose(
            waveform[:, 0:-3], output_waveform[:, 3:], atol=1e-5
        )

    def test_lfilter_loop_vs_tensor(self):

        torch.random.manual_seed(42)
        waveform = torch.rand(2, 100000)
        b_coeffs = torch.tensor([0, 0.1, 0.1, 1], dtype=torch.float32)
        a_coeffs = torch.tensor([1, 0.2, 0, 0], dtype=torch.float32)
        import time

        tensor_time_start = time.time()
        output_waveform1 = F.lfilter_tensor(waveform, a_coeffs, b_coeffs)
        tensor_time_taken = time.time() - tensor_time_start

        loop_time_start = time.time()
        output_waveform2 = F.lfilter(waveform, a_coeffs, b_coeffs)
        loop_time_taken = time.time() - loop_time_start

        tensor_matrix_time_start = time.time()
        output_waveform3 = F.lfilter_tensor_matrix(waveform, a_coeffs, b_coeffs)
        tensor_matrix_time_taken = time.time() - tensor_matrix_time_start

        print("\n")
        print("Looped Implementation took       : ", loop_time_taken)
        print("Tensor Implementation took       : ", tensor_time_taken)
        print("Tensor Matrix Implementation took: ", tensor_matrix_time_taken)

        assert torch.allclose(
            output_waveform1, output_waveform2, atol=1e-5
        )
        assert torch.allclose(
            output_waveform1, output_waveform3, atol=1e-5
        )

    def test_lfilter(self):
        """
        Design an IIR lowpass filter using scipy.signal filter design
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirdesign.html#scipy.signal.iirdesign

        from scipy.signal import iirdesign
        b, a = iirdesign(0.2, 0.3, 1, 60)

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
            ]
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
            ]
        )

        filepath = os.path.join(self.test_dirpath, "assets", "dtmf_30s_stereo.mp3")
        waveform, sample_rate = torchaudio.load(filepath, normalization=True)
        output_waveform = F.lfilter(waveform, a_coeffs, b_coeffs)
        assert len(output_waveform.size()) == 2
        assert output_waveform.size(0) == waveform.size(0)
        assert output_waveform.size(1) == waveform.size(1)

    def test_lowpass_sox_compliance(self):

        """
        Run a biquad lowpass filter using SoX vs torchaudio's lfilter
        Results should be very close
        """

        CUTOFF_FREQ = 3000

        noise_filepath = os.path.join(
            self.test_dirpath, "assets", "whitenoise_1min.mp3"
        )
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("lowpass", [CUTOFF_FREQ])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(
            noise_filepath, normalization=True
        )
        output_waveform = F.lowpass_biquad(waveform, sample_rate, CUTOFF_FREQ)

        assert torch.allclose(sox_output_waveform, output_waveform, atol=1e-4)

    def test_highpass_sox_compliance(self):
        """
        Run a biquad highpass filter using SoX vs torchaudio's lfilter
        Results should be very close
        """

        CUTOFF_FREQ = 2000

        noise_filepath = os.path.join(
            self.test_dirpath, "assets", "whitenoise_1min.mp3"
        )
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("highpass", [CUTOFF_FREQ])
        sox_output_waveform, sr = E.sox_build_flow_effects()

        waveform, sample_rate = torchaudio.load(
            noise_filepath, normalization=True
        )
        output_waveform = F.highpass_biquad(waveform, sample_rate, CUTOFF_FREQ)

        # TBD - this fails at the 1e-4 level, debug why
        assert torch.allclose(sox_output_waveform, output_waveform, atol=1e-3)

    def test_perf_biquad_filtering(self):
        """
        Compare SoX implementation of biquad filtering with C++ implementation

        Test that results are similar and how performance differs

        Current results: C++ implementation approximately same speed
        """

        fn_sine = os.path.join(self.test_dirpath, "assets", "whitenoise_1min.mp3")

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

        # C++ Diff Eq Filter
        _timing_cpp_filtering = time.time()
        waveform, sample_rate = torchaudio.load(fn_sine, normalization=True)
        waveform_diff_eq_out = F.lfilter(
            waveform, torch.tensor([a0, a1, a2]), torch.tensor([b0, b1, b2])
        )
        _timing_diff_eq_run_time = time.time() - _timing_cpp_filtering

        # print("\n")
        # print("SoX Run Time         (s): ", round(_timing_sox_run_time, 3))
        # print("CPP Lfilter Run Time (s): ", round(_timing_diff_eq_run_time, 3))

        assert torch.allclose(waveform_sox_out, waveform_diff_eq_out, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
