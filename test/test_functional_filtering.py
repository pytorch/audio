from __future__ import absolute_import, division, print_function, unicode_literals
import math
import os
import torch
import torchaudio
import torchaudio.functional as F
import unittest
import common_utils


class TestFunctionalFiltering(unittest.TestCase):
    test_dirpath, test_dir = common_utils.create_temp_assets_dir()

    """
    Open up a file of white noise.
    Compute ratio of power in high band vs low bands, should be approximately equal
    Perform a low pass filter at half Nyquist
    Recompute ratio, should be very skewed
    """

    @unittest.skip
    def test_band_energy_reduction(self):
        def compute_ratio_high_vs_low(signal, n_fft):
            dft = torch.stft(signal, n_fft)
            psd = torch.sqrt(
                dft[:, :, :, 0] * dft[:, :, :, 0] + dft[:, :, :, 1] * dft[:, :, :, 1]
            )
            upper_indices = [
                slice(None),
                torch.arange(n_fft // 4, n_fft // 2 + 1).long(),
                slice(None),
            ]
            lower_indices = [
                slice(None),
                torch.arange(0, n_fft // 4).long(),
                slice(None),
            ]

            higher_bands_avg_energy = torch.mean(psd[upper_indices]).item()
            lower_bands_avg_energy = torch.mean(psd[lower_indices]).item()
            return higher_bands_avg_energy / lower_bands_avg_energy

        white_noise_path = os.path.join(
            os.path.join(self.test_dirpath, "assets"), "whitenoise.mp3"
        )
        audio, sample_rate = torchaudio.load(white_noise_path, normalization=False)
        high_to_low_pre_lowpass = compute_ratio_high_vs_low(audio, 512)
        self.assertTrue(0.25 < high_to_low_pre_lowpass < 4)
        CUTOFF_FREQ = sample_rate // 4  # half of nyquist
        lowpassed_audio = F.lowpass(audio, sample_rate, 512, CUTOFF_FREQ)
        high_to_low_post_lowpass = compute_ratio_high_vs_low(lowpassed_audio, 512)

        print(high_to_low_pre_lowpass, high_to_low_post_lowpass)
        self.assertTrue(high_to_low_post_lowpass * 10 < high_to_low_pre_lowpass)

    @unittest.skip        
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
        signal_out = F.lowpass_biquad(noise, sample_rate, CUTOFF_FREQ)

        assert torch.allclose(sox_signal_out, signal_out, atol=1e-4)

    def test_biquad_filtering(self):
        """
        Try three implementations of biquad filtering and test speed
        - 1.) Call to SoX
        - 2.) Call to cpp biquad function
        - 3.) Call to Python biquad function

        Current results: 1 ~20x faster than 2 ~ 15x faster than 3
        """

        import time, _torch_filtering

        b0 = 0.4
        b1 = 0.2
        b2 = 0.9
        a0 = 0.7
        a1 = 0.2
        a2 = 0.6

        # SoX method
        fn_sine = os.path.join(self.test_dirpath, "assets", "sinewave.wav")
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(fn_sine)
        _timing_sox = time.time()
        E.append_effect_to_chain("biquad", [b0, b1, b2, a0, a1, a2])
        waveform_sox_out, sr = E.sox_build_flow_effects()
        _timing_sox_run_time = time.time() - _timing_sox
        
        # CPP Filtering with Biquad
        audio, sample_rate = torchaudio.load(fn_sine, normalization=True)
        waveform_cpp_out = torch.zeros_like(audio)
        _timing_cpp_filtering = time.time()
        _torch_filtering.biquad(audio, waveform_cpp_out, b0, b1, b2, a0, a1, a2)
        _timing_cpp_run_time = time.time() - _timing_cpp_filtering

        # Native Python Implementation
        _timing_python = time.time()
        waveform_python_out = F.biquad(audio, b0, b1, b2, a0, a1, a2)
        _timing_python_run_time = time.time() - _timing_python        
        
        assert torch.allclose(waveform_sox_out, waveform_python_out, atol=1e-5)
        assert torch.allclose(waveform_sox_out, waveform_cpp_out, atol=1e-5)

        print("SoX Run Time   : ", round(_timing_sox_run_time, 3))
        print("CPP Run Time   : ", round(_timing_cpp_run_time, 3))
        print("Python Run Time: ", round(_timing_python_run_time, 3))

        assert( 1== 0)


    @unittest.skip
    def test_highpass_sox_compliance(self):

        """
        Run a file through SoX highpass filter
        Then run through our highpass filter
        Should match
        """

        CUTOFF_FREQ = 1000

        noise_filepath = os.path.join(self.test_dirpath, "assets", "whitenoise.mp3")
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(noise_filepath)
        E.append_effect_to_chain("highpass", [CUTOFF_FREQ])
        sox_signal_out, sr = E.sox_build_flow_effects()

        noise, sample_rate = torchaudio.load(noise_filepath, normalization=True)
        signal_out = F.highpass_biquad(noise, sample_rate, CUTOFF_FREQ)

        assert torch.allclose(sox_signal_out, signal_out, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
