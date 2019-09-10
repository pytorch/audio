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
    def test_band_energy_reduction(self):

        def compute_ratio_high_vs_low(signal, n_fft):
            dft = torch.stft(signal, n_fft)
            psd = torch.sqrt(dft[:,:,:,0] * dft[:,:,:,0] + dft[:,:,:,1] * dft[:,:,:,1])
            upper_indices = [slice(None), torch.arange(n_fft//4, n_fft//2+1).long(), slice(None)]
            lower_indices = [slice(None), torch.arange(0, n_fft//4).long(), slice(None)]

            higher_bands_avg_energy = torch.mean(psd[upper_indices]).item()
            lower_bands_avg_energy = torch.mean(psd[lower_indices]).item()
            return (higher_bands_avg_energy / lower_bands_avg_energy)

        white_noise_path = os.path.join(os.path.join(self.test_dirpath, "assets"), 'whitenoise.mp3')
        audio, sample_rate = torchaudio.load(white_noise_path, normalization=False)
        high_to_low_pre_lowpass = compute_ratio_high_vs_low(audio, 512)
        self.assertTrue(0.25 < high_to_low_pre_lowpass < 4)
        CUTOFF_FREQ = sample_rate // 4 # half of nyquist
        lowpassed_audio = torchaudio.functional.lowpass(audio, sample_rate, 512, CUTOFF_FREQ)
        high_to_low_post_lowpass = compute_ratio_high_vs_low(lowpassed_audio, 512)
        
        print(high_to_low_pre_lowpass, high_to_low_post_lowpass)
        self.assertTrue( high_to_low_post_lowpass * 10 < high_to_low_pre_lowpass )

    def test_biquad_sox_compliance(self):

        """
        Run a file through SoX biquad with randomly chosen coefficients
        Then run a file through our biquad method. 
        Should match
        """
        b0 = 0.4
        b1 = 0.2
        b2 = 0.9
        a0 = 0.7
        a1 = 0.2
        a2 = 0.6

        fn_sine = os.path.join(self.test_dirpath, "assets", "sinewave.wav")
        
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(fn_sine)
        E.append_effect_to_chain("biquad", [b0, b1, b2, a0, a1, a2])
        sox_signal_out, sr = E.sox_build_flow_effects()
        
        audio, sample_rate = torchaudio.load(fn_sine, normalization=True)
        signal_out = torchaudio.functional.biquad(audio, b0, b1, b2, a0, a1, a2)

        assert(torch.allclose(sox_signal_out, signal_out, atol=1e-4))

if __name__ == '__main__':
    unittest.main()
