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

    def test_lowpass_sox_compliance(self):

        """
        Run a file through SoX lowpass filter
        Then run through our lowpass filter
        Should match
        """

        CUTOFF_FREQ = 3000

        noise_filepath = os.path.join(self.test_dirpath, "assets", "whitenoise.mp3")
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("lowpass", [CUTOFF_FREQ])
        sox_signal_out, sr = E.sox_build_flow_effects()

        noise, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        signal_out = F.lowpass_biquad_python(noise, sample_rate, CUTOFF_FREQ)

        assert torch.allclose(sox_signal_out, signal_out, atol=1e-4)

    def test_diff_eq(self):
        """
        Design a lowpass filter using scipy.signal filter design
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirdesign.html#scipy.signal.iirdesign

        from scipy.signal import iirdesign
        b, a = iirdesign(0.2, 0.3, 1, 60)

        Test is that this runs
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

        noise_filepath = os.path.join(self.test_dirpath, "assets", "whitenoise.mp3")
        audio, sample_rate = torchaudio.load(noise_filepath, normalization=True)

        waveform_diff_eq_out = torch.zeros_like(audio)
        F.diffeq_cpp(audio, waveform_diff_eq_out, a_coeffs, b_coeffs)

    def test_low_pass_perf_iir_vs_fir(self):
        """
        Try two low pass filters to compare performance
        - 10th order IIR 
        - 40th order FIR
        """

        fir_impulse_response = [
            -0.00040463742194593165,
            -0.0011997613323707066,
            -0.0018908492870125047,
            -0.00207654308585229,
            -0.001096110046662624,
            0.0015056218536868779,
            0.005330573111949219,
            0.008753080180802494,
            0.009251376850303583,
            0.004551586188930009,
            -0.00579865949432587,
            -0.01919960191734472,
            -0.029922330203954777,
            -0.03058886187399114,
            -0.014900805267782886,
            0.019382291456268037,
            0.06852266400807679,
            0.12297037769851049,
            0.16984408705548223,
            0.19696650152723372,
            0.19696650152723374,
            0.16984408705548223,
            0.12297037769851049,
            0.06852266400807679,
            0.01938229145626804,
            -0.014900805267782887,
            -0.03058886187399115,
            -0.029922330203954784,
            -0.01919960191734472,
            -0.00579865949432587,
            0.004551586188930011,
            0.009251376850303588,
            0.008753080180802494,
            0.005330573111949219,
            0.0015056218536868779,
            -0.001096110046662626,
            -0.0020765430858522907,
            -0.0018908492870125056,
            -0.0011997613323707066,
            -0.00040463742194593165,
        ]

        iir_b_coeffs = torch.tensor(
            [
                4.14176942e-05,
                -6.74168865e-05,
                1.28894969e-04,
                -7.60724730e-05,
                6.18923831e-05,
                6.18923831e-05,
                -7.60724730e-05,
                1.28894969e-04,
                -6.74168865e-05,
                4.14176942e-05,
            ]
        )
        iir_a_coeffs = torch.tensor(
            [
                1.0,
                -7.51701965,
                25.93753044,
                -53.79433042,
                73.79210968,
                -69.36257588,
                44.65306696,
                -18.98130961,
                4.83553413,
                -0.56282822,
            ]
        )

        noise_filepath = os.path.join(self.test_dirpath, "assets", "whitenoise.mp3")
        noise, sample_rate = torchaudio.load(noise_filepath, normalization=True)

        waveform_diff_eq_out = torch.zeros_like(noise)
        _diffeq_start_time = time.time()
        F.diffeq_cpp(noise, waveform_diff_eq_out, iir_a_coeffs, iir_b_coeffs)
        _diffeq_run_time = time.time() - _diffeq_start_time

        # TBD

        print("Diff Eq Low Pass Run Time:", _diffeq_run_time)
        # print("FIR Low Pass Run Time    :", _fir_run_time)

    def test_perf_biquad_filtering(self):
        """
        Try implementations of biquad filtering and test speed
        - 1.) Call to SoX
        - 2.) Call to cpp biquad function
        - 3.) Call to cpp general difference equation function
        - 4.) Call to Python biquad function

        Current results: 1 ~20x faster than 2, 2 ~ 2x faster than 3, 3 ~8x faster than 4
        """

        fn_sine = os.path.join(self.test_dirpath, "assets", "sinewave.wav")
        audio, sample_rate = torchaudio.load(fn_sine, normalization=True)

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

        # CPP Filtering with Biquad
        waveform_cpp_out = torch.zeros_like(audio)
        _timing_cpp_filtering = time.time()
        F.biquad_cpp(audio, waveform_cpp_out, b0, b1, b2, a0, a1, a2)
        _timing_cpp_run_time = time.time() - _timing_cpp_filtering

        # Diff Eq Filter
        waveform_diff_eq_out = torch.zeros_like(audio)
        _timing_cpp_filtering = time.time()
        F.diffeq_cpp(
            audio,
            waveform_diff_eq_out,
            torch.tensor([a0, a1, a2]),
            torch.tensor([b0, b1, b2]),
        )
        _timing_diff_eq_run_time = time.time() - _timing_cpp_filtering

        # Native Python Implementation
        _timing_python = time.time()
        waveform_python_out = F.biquad_python(audio, b0, b1, b2, a0, a1, a2)
        _timing_python_run_time = time.time() - _timing_python

        assert torch.allclose(waveform_sox_out, waveform_python_out, atol=1e-5)
        assert torch.allclose(waveform_sox_out, waveform_cpp_out, atol=1e-5)
        assert torch.allclose(waveform_sox_out, waveform_diff_eq_out, atol=1e-5)

        print("SoX Run Time          : ", round(_timing_sox_run_time, 3))
        print("CPP Biquad Run Time   : ", round(_timing_cpp_run_time, 3))
        print("CPP Diff Eq Run Time  : ", round(_timing_diff_eq_run_time, 3))
        print("Python Run Time       : ", round(_timing_python_run_time, 3))


if __name__ == "__main__":
    unittest.main()
