import unittest
import torch
import torchaudio
import math
import os


class Test_LoadSave(unittest.TestCase):
    test_dirpath = os.path.dirname(os.path.realpath(__file__))
    test_filepath = os.path.join(test_dirpath, "assets",
                                 "steam-train-whistle-daniel_simon.mp3")

    def test_1_save(self):
        # load signal
        x, sr = torchaudio.load(self.test_filepath, normalization=False)

        # check save
        new_filepath = os.path.join(self.test_dirpath, "test.wav")
        torchaudio.save(new_filepath, x, sr)
        self.assertTrue(os.path.isfile(new_filepath))
        os.unlink(new_filepath)

        # check automatic normalization
        x /= 1 << 31
        torchaudio.save(new_filepath, x, sr)
        self.assertTrue(os.path.isfile(new_filepath))
        os.unlink(new_filepath)

        # test save 1d tensor
        x = x[0, :]  # get mono signal
        x.squeeze_()  # remove channel dim
        torchaudio.save(new_filepath, x, sr)
        self.assertTrue(os.path.isfile(new_filepath))
        os.unlink(new_filepath)

        # don't allow invalid sizes as inputs
        with self.assertRaises(ValueError):
            x.unsqueeze_(1)  # L x C not C x L
            torchaudio.save(new_filepath, x, sr)

        with self.assertRaises(ValueError):
            x.squeeze_()
            x.unsqueeze_(1)
            x.unsqueeze_(0)  # 1 x L x 1
            torchaudio.save(new_filepath, x, sr)

        # don't save to folders that don't exist
        with self.assertRaises(OSError):
            new_filepath = os.path.join(self.test_dirpath, "no-path",
                                        "test.wav")
            torchaudio.save(new_filepath, x, sr)

        # save created file
        sinewave_filepath = os.path.join(self.test_dirpath, "assets",
                                         "sinewave.wav")
        sr = 16000
        freq = 440
        volume = 0.3

        y = (torch.cos(
            2 * math.pi * torch.arange(0, 4 * sr).float() * freq / sr))
        y.unsqueeze_(0)
        # y is between -1 and 1, so must scale
        y = (y * volume * (2**31)).long()
        torchaudio.save(sinewave_filepath, y, sr)
        self.assertTrue(os.path.isfile(sinewave_filepath))

        # test precision
        new_precision = 32
        new_filepath = os.path.join(self.test_dirpath, "test.wav")
        si, ei = torchaudio.info(sinewave_filepath)
        torchaudio.save(new_filepath, y, sr, new_precision)
        si32, ei32 = torchaudio.info(new_filepath)
        self.assertEqual(si.precision, 16)
        self.assertEqual(si32.precision, new_precision)
        os.unlink(new_filepath)

    def test_2_load(self):
        # check normal loading
        x, sr = torchaudio.load(self.test_filepath)
        self.assertEqual(sr, 44100)
        self.assertEqual(x.size(), (2, 278756))

        # check no normalizing
        x, _ = torchaudio.load(self.test_filepath, normalization=False)
        self.assertTrue(x.min() <= -1.0)
        self.assertTrue(x.max() >= 1.0)

        # check offset
        offset = 15
        x, _ = torchaudio.load(self.test_filepath)
        x_offset, _ = torchaudio.load(self.test_filepath, offset=offset)
        self.assertTrue(x[:, offset:].allclose(x_offset))

        # check number of frames
        n = 201
        x, _ = torchaudio.load(self.test_filepath, num_frames=n)
        self.assertTrue(x.size(), (2, n))

        # check channels first
        x, _ = torchaudio.load(self.test_filepath, channels_first=False)
        self.assertEqual(x.size(), (278756, 2))

        # check different input tensor type
        x, _ = torchaudio.load(self.test_filepath, torch.LongTensor(), normalization=False)
        self.assertTrue(isinstance(x, torch.LongTensor))

        # check raising errors
        with self.assertRaises(OSError):
            torchaudio.load("file-does-not-exist.mp3")

        with self.assertRaises(OSError):
            tdir = os.path.join(
                os.path.dirname(self.test_dirpath), "torchaudio")
            torchaudio.load(tdir)

    def test_3_load_and_save_is_identity(self):
        input_path = os.path.join(self.test_dirpath, 'assets', 'sinewave.wav')
        tensor, sample_rate = torchaudio.load(input_path)
        output_path = os.path.join(self.test_dirpath, 'test.wav')
        torchaudio.save(output_path, tensor, sample_rate)
        tensor2, sample_rate2 = torchaudio.load(output_path)
        self.assertTrue(tensor.allclose(tensor2))
        self.assertEqual(sample_rate, sample_rate2)
        os.unlink(output_path)

    def test_4_load_partial(self):
        num_frames = 101
        offset = 201
        # load entire mono sinewave wav file, load a partial copy and then compare
        input_sine_path = os.path.join(self.test_dirpath, 'assets', 'sinewave.wav')
        x_sine_full, sr_sine = torchaudio.load(input_sine_path)
        x_sine_part, _ = torchaudio.load(input_sine_path, num_frames=num_frames, offset=offset)
        l1_error = x_sine_full[:, offset:(num_frames + offset)].sub(x_sine_part).abs().sum().item()
        # test for the correct number of samples and that the correct portion was loaded
        self.assertEqual(x_sine_part.size(1), num_frames)
        self.assertEqual(l1_error, 0.)
        # create a two channel version of this wavefile
        x_2ch_sine = x_sine_full.repeat(1, 2)
        out_2ch_sine_path = os.path.join(self.test_dirpath, 'assets', '2ch_sinewave.wav')
        torchaudio.save(out_2ch_sine_path, x_2ch_sine, sr_sine)
        x_2ch_sine_load, _ = torchaudio.load(out_2ch_sine_path, num_frames=num_frames, offset=offset)
        os.unlink(out_2ch_sine_path)
        l1_error = x_2ch_sine_load.sub(x_2ch_sine[:, offset:(offset + num_frames)]).abs().sum().item()
        self.assertEqual(l1_error, 0.)

        # test with two channel mp3
        x_2ch_full, sr_2ch = torchaudio.load(self.test_filepath, normalization=True)
        x_2ch_part, _ = torchaudio.load(self.test_filepath, normalization=True, num_frames=num_frames, offset=offset)
        l1_error = x_2ch_full[:, offset:(offset + num_frames)].sub(x_2ch_part).abs().sum().item()
        self.assertEqual(x_2ch_part.size(1), num_frames)
        self.assertEqual(l1_error, 0.)

        # check behavior if number of samples would exceed file length
        offset_ns = 300
        x_ns, _ = torchaudio.load(input_sine_path, num_frames=100000, offset=offset_ns)
        self.assertEqual(x_ns.size(1), x_sine_full.size(1) - offset_ns)

        # check when offset is beyond the end of the file
        with self.assertRaises(RuntimeError):
            torchaudio.load(input_sine_path, offset=100000)

    def test_5_get_info(self):
        input_path = os.path.join(self.test_dirpath, 'assets', 'sinewave.wav')
        channels, samples, rate, precision = (1, 64000, 16000, 16)
        si, ei = torchaudio.info(input_path)
        self.assertEqual(si.channels, channels)
        self.assertEqual(si.length, samples)
        self.assertEqual(si.rate, rate)
        self.assertEqual(ei.bits_per_sample, precision)

if __name__ == '__main__':
    unittest.main()
