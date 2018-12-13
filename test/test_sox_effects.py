import unittest
import torch
import torchaudio
import math
import os


class Test_SoxEffectsChain(unittest.TestCase):
    test_dirpath = os.path.dirname(os.path.realpath(__file__))
    test_filepath = os.path.join(test_dirpath, "assets",
                                 "steam-train-whistle-daniel_simon.mp3")

    def test_single_channel(self):
        fn_sine = os.path.join(self.test_dirpath, "assets", "sinewave.wav")
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(fn_sine)
        E.append_effect_to_chain("echos", [0.8, 0.7, 40, 0.25, 63, 0.3])
        x, sr = E.sox_build_flow_effects()
        # check if effects worked
        #print(x.size())

    def test_rate_channels(self):
        target_rate = 16000
        target_channels = 1
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(self.test_filepath)
        E.append_effect_to_chain("rate", [target_rate])
        E.append_effect_to_chain("channels", [target_channels])
        x, sr = E.sox_build_flow_effects()
        # check if effects worked
        self.assertEqual(sr, target_rate)
        self.assertEqual(x.size(0), target_channels)

    def test_lowpass_speed(self):
        speed = .8
        si, _ = torchaudio.info(self.test_filepath)
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(self.test_filepath)
        E.append_effect_to_chain("lowpass", 100)
        E.append_effect_to_chain("speed", speed)
        E.append_effect_to_chain("rate", si.rate)
        x, sr = E.sox_build_flow_effects()
        # check if effects worked
        self.assertEqual(x.size(1), int((si.length / si.channels) / speed))

    def test_ulaw_and_siginfo(self):
        si_out = torchaudio.sox_signalinfo_t()
        ei_out = torchaudio.sox_encodinginfo_t()
        si_out.precision = 8
        ei_out.encoding = torchaudio.get_sox_encoding_t(9)
        ei_out.bits_per_sample = 8
        si_in, ei_in = torchaudio.info(self.test_filepath)
        si_out.rate = 44100
        si_out.channels = 2
        E = torchaudio.sox_effects.SoxEffectsChain(out_siginfo=si_out, out_encinfo=ei_out)
        E.set_input_file(self.test_filepath)
        x, sr = E.sox_build_flow_effects()
        # Note: the output was encoded into ulaw because the
        #       number of unique values in the output is less than 256.
        self.assertLess(x.unique().size(0), 2**8)
        self.assertEqual(x.numel(), si_in.length)

    def test_band_chorus(self):
        si_in, ei_in = torchaudio.info(self.test_filepath)
        ei_in.encoding = torchaudio.get_sox_encoding_t(1)
        E = torchaudio.sox_effects.SoxEffectsChain(out_encinfo=ei_in, out_siginfo=si_in)
        E.set_input_file(self.test_filepath)
        E.append_effect_to_chain("band", ["-n", "10k", "3.5k"])
        E.append_effect_to_chain("chorus", [.5, .7, 55, 0.4, .25, 2, '-s'])
        E.append_effect_to_chain("rate", [si_in.rate])
        E.append_effect_to_chain("channels", [si_in.channels])
        x, sr = E.sox_build_flow_effects()
        # The chorus effect will make the output file longer than the input
        self.assertEqual(x.size(0), si_in.channels)
        self.assertGreaterEqual(x.size(1) * x.size(0), si_in.length)

    def test_synth(self):
        si_in, ei_in = torchaudio.info(self.test_filepath)
        len_in_seconds = si_in.length / si_in.channels / si_in.rate
        ei_in.encoding = torchaudio.get_sox_encoding_t(1)
        E = torchaudio.sox_effects.SoxEffectsChain(out_encinfo=ei_in, out_siginfo=si_in)
        E.set_input_file(self.test_filepath)
        E.append_effect_to_chain("synth", [str(len_in_seconds), "pinknoise", "mix"])
        E.append_effect_to_chain("rate", [44100])
        E.append_effect_to_chain("channels", [2])
        x, sr = E.sox_build_flow_effects()
        self.assertEqual(x.size(0), si_in.channels)
        self.assertEqual(si_in.length, x.size(0) * x.size(1))

    def test_gain(self):
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(self.test_filepath)
        E.append_effect_to_chain("gain", ["5"])
        x, sr = E.sox_build_flow_effects()
        E.clear_chain()
        self.assertTrue(x.abs().max().item(), 1.)
        E.set_input_file(self.test_filepath)
        E.append_effect_to_chain("gain", ["-e", "-5"])
        x, sr = E.sox_build_flow_effects()
        E.clear_chain()
        self.assertLess(x.abs().max().item(), 1.)
        E.set_input_file(self.test_filepath)
        E.append_effect_to_chain("gain", ["-b", "8"])
        x, sr = E.sox_build_flow_effects()
        E.clear_chain()
        self.assertTrue(x.abs().max().item(), 1.)
        E.set_input_file(self.test_filepath)
        E.append_effect_to_chain("gain", ["-n", "-10"])
        x, sr = E.sox_build_flow_effects()
        E.clear_chain()
        self.assertLess(x.abs().max().item(), 1.)

    def test_tempo(self):
        tempo = .8
        si, _ = torchaudio.info(self.test_filepath)
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(self.test_filepath)
        E.append_effect_to_chain("tempo", ["-s", tempo])
        x, sr = E.sox_build_flow_effects()
        # check if effect worked
        self.assertEqual(x.size(1), int((si.length / si.channels) / tempo))

    def test_trim(self):
        x_orig, _ = torchaudio.load(self.test_filepath)
        offset = "10000s"
        offset_int = int(offset[:-1])
        num_frames = "20000s"
        num_frames_int = int(num_frames[:-1])
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(self.test_filepath)
        E.append_effect_to_chain("trim", [offset, num_frames])
        x, sr = E.sox_build_flow_effects()
        # check if effect worked
        self.assertTrue(x.allclose(x_orig[:,offset_int:(offset_int+num_frames_int)], rtol=1e-4, atol=1e-4))

    def test_silence_contrast(self):
        si, _ = torchaudio.info(self.test_filepath)
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(self.test_filepath)
        E.append_effect_to_chain("silence", [1, 100, 1])
        E.append_effect_to_chain("contrast", [])
        x, sr = E.sox_build_flow_effects()
        # check if effect worked
        self.assertLess(x.numel(), si.length)

    def test_reverse(self):
        x_orig, _ = torchaudio.load(self.test_filepath)
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(self.test_filepath)
        E.append_effect_to_chain("reverse", "")
        x_rev, _ = E.sox_build_flow_effects()
        # check if effect worked
        rev_idx = torch.LongTensor(range(x_orig.size(1))[::-1])
        self.assertTrue(x_orig.allclose(x_rev[:, rev_idx], rtol=1e-5, atol=2e-5))

    def test_compand_fade(self):
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(self.test_filepath)
        E.append_effect_to_chain("compand", ["0.3,1", "6:-70,-60,-20", "-5", "-90", "0.2"])
        E.append_effect_to_chain("fade", ["q", "0.25", "0", "0.33"])
        x, _ = E.sox_build_flow_effects()
        # check if effect worked
        #print(x.size())

    def test_biquad_delay(self):
        si, _ = torchaudio.info(self.test_filepath)
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(self.test_filepath)
        E.append_effect_to_chain("biquad", ["0.25136437", "0.50272873", "0.25136437", "1.0", "-0.17123075", "0.17668821"])
        E.append_effect_to_chain("delay", ["15000s"])
        x, _ = E.sox_build_flow_effects()
        # check if effect worked
        self.assertTrue(x.size(1) == (si.length / si.channels) + 15000)

    def test_invalid_effect_name(self):
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(self.test_filepath)
        # there is no effect named "special"
        with self.assertRaises(LookupError):
            E.append_effect_to_chain("special", [""])

    def test_unimplemented_effect(self):
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(self.test_filepath)
        # the sox spectrogram function is not implemented in torchaudio
        with self.assertRaises(NotImplementedError):
            E.append_effect_to_chain("spectrogram", [""])

    def test_invalid_effect_options(self):
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(self.test_filepath)
        # first two options should be combined to "0.3,1"
        E.append_effect_to_chain("compand", ["0.3", "1", "6:-70,-60,-20", "-5", "-90", "0.2"])
        with self.assertRaises(RuntimeError):
            E.sox_build_flow_effects()


if __name__ == '__main__':
    torchaudio.initialize_sox()
    unittest.main()
    torchaudio.shutdown_sox()
