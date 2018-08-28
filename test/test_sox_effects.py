import unittest
import torch
import torchaudio
import math
import os


class Test_SoxEffects(unittest.TestCase):
    test_dirpath = os.path.dirname(os.path.realpath(__file__))
    test_filepath = os.path.join(test_dirpath, "assets",
                                 "steam-train-whistle-daniel_simon.mp3")

    def test_rate_channels(self):
        target_rate = 16000
        target_channels = 1
        E = torchaudio.sox_effects.SoxEffects()
        E.set_input_file(self.test_filepath)
        E.sox_append_effect_to_chain("rate", [target_rate])
        E.sox_append_effect_to_chain("channels", [target_channels])
        x, sr = E.sox_build_flow_effects()
        # check if effects worked
        self.assertEqual(sr, target_rate)
        self.assertEqual(x.size(0), target_channels)

    def test_other(self):
        speed = .8
        si, _ = torchaudio.info(self.test_filepath)
        E = torchaudio.sox_effects.SoxEffects()
        E.set_input_file(self.test_filepath)
        E.sox_append_effect_to_chain("lowpass", 100)
        E.sox_append_effect_to_chain("speed", speed)
        E.sox_append_effect_to_chain("rate", si.rate)
        x, sr = E.sox_build_flow_effects()
        # check if effects worked
        self.assertEqual(x.size(1), int((si.length / si.channels) / speed))

    def test_ulaw_and_siginfo(self):
        si_out = torchaudio.sox_signalinfo_t()
        ei_out = torchaudio.sox_encodinginfo_t()
        si_out.rate = 16000
        si_out.channels = 1
        si_out.precision = 8
        ei_out.encoding = torchaudio.get_sox_encoding_t(9)
        ei_out.bits_per_sample = 8
        si_in, ei_in = torchaudio.info(self.test_filepath)
        E = torchaudio.sox_effects.SoxEffects(out_siginfo=si_out, out_encinfo=ei_out)
        E.set_input_file(self.test_filepath)
        x, sr = E.sox_build_flow_effects()
        # Note: the sample rate is reported as "changed", but no downsampling occured
        #       also the number of channels has not changed.  Run rate and channels effects
        #       to make those changes
        self.assertLess(x.unique().size(0), 2**8)
        self.assertEqual(x.size(0), si_in.channels)
        self.assertEqual(sr, si_out.rate)
        self.assertEqual(x.numel(), si_in.length)

if __name__ == '__main__':
    torchaudio.initialize_sox()
    unittest.main()
    torchaudio.shutdown_sox()
