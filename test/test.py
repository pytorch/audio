import unittest
import torch
import torchaudio
import math
import os

class Test_LoadSave(unittest.TestCase):
    test_dirpath = os.path.dirname(os.path.realpath(__file__))
    test_mp3_filepath = os.path.join(test_dirpath, "assets", "steam-train-whistle-daniel_simon.mp3")
    test_wav_filepath = os.path.join(test_dirpath, "assets", "16kwav.wav")

    def test_get_info(self):
        info = torchaudio.get_info(self.test_mp3_filepath)
        self.assertEqual(info, {'bits_per_sample': 16, 'nframes': 278756, 'sample_rate': 44100, 'nchannels': 2})

        info = torchaudio.get_info(self.test_wav_filepath)
        self.assertEqual(info, {'bits_per_sample': 16, 'nframes': 320000, 'sample_rate': 16000, 'nchannels': 1})

    def test_load(self):
        # check normal loading
        x, sr = torchaudio.load(self.test_mp3_filepath)
        self.assertEqual(sr, 44100)
        self.assertEqual(x.size(), (278756,2))

        # check offset and nframes

        whole_audio, sr = torchaudio.load(self.test_wav_filepath)
        self.assertEqual(sr, 16000)
        self.assertEqual(whole_audio.size(), (320000, 1))

        x, sr = torchaudio.load(self.test_wav_filepath, offset=100)
        self.assertEqual(x.size(), (320000 - 100, 1))

        x, sr = torchaudio.load(self.test_wav_filepath, offset=100, nframes=100)
        self.assertEqual(x.size(), (100, 1))

        x, sr = torchaudio.load(self.test_wav_filepath, offset=319999, nframes=100)
        self.assertEqual(x.size(), (1, 1))

        x, sr = torchaudio.load(self.test_wav_filepath, offset=320000, nframes=100)
        # Maybe it would be better if we returned a Tensor of size (0,1), however PyTorch in general doesn't handle
        # Tensors where one of the dimensions is 0 as one might expect, so not sure if it would really help to have
        # different behavior
        self.assertEqual(x.size(), ())

        with self.assertRaises(IndexError):
            x, sr = torchaudio.load(self.test_wav_filepath, offset=320001, nframes=100)

        # check scaling
        x, sr = torchaudio.load(self.test_mp3_filepath)
        self.assertTrue(x.min() >= -1.0)
        self.assertTrue(x.max() <= 1.0)

        # check raising errors
        with self.assertRaises(OSError):
            torchaudio.load("file-does-not-exist.mp3")

        with self.assertRaises(OSError):
            tdir = os.path.join(os.path.dirname(self.test_dirpath), "torchaudio")
            torchaudio.load(tdir)

    def test_save(self):
        # load signal
        x, sr = torchaudio.load(self.test_mp3_filepath)

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
        x = x[:, 0] # get mono signal
        x.squeeze_() # remove channel dim
        torchaudio.save(new_filepath, x, sr)
        self.assertTrue(os.path.isfile(new_filepath))
        os.unlink(new_filepath)

        # don't allow invalid sizes as inputs
        with self.assertRaises(ValueError):
            x.unsqueeze_(0) # N x L not L x N
            torchaudio.save(new_filepath, x, sr)

        with self.assertRaises(ValueError):
            x.squeeze_()
            x.unsqueeze_(1)
            x.unsqueeze_(0) # 1 x L x 1
            torchaudio.save(new_filepath, x, sr)

        # automatically convert sr from floating point to int
        x.squeeze_(0)
        torchaudio.save(new_filepath, x, float(sr))
        self.assertTrue(os.path.isfile(new_filepath))
        os.unlink(new_filepath)

        # don't allow uneven integers
        with self.assertRaises(TypeError):
            torchaudio.save(new_filepath, x, float(sr) + 0.5)
            self.assertTrue(os.path.isfile(new_filepath))
            os.unlink(new_filepath)

        # don't save to folders that don't exist
        with self.assertRaises(OSError):
            new_filepath = os.path.join(self.test_dirpath, "no-path", "test.wav")
            torchaudio.save(new_filepath, x, sr)

        # save created file
        sinewave_filepath = os.path.join(self.test_dirpath, "assets", "sinewave.wav")
        sr = 16000
        freq = 440
        volume = 0.3

        y = (torch.cos(2*math.pi*torch.arange(0, 4*sr) * freq/sr)).float()
        y.unsqueeze_(1)
        # y is between -1 and 1, so must scale
        y = (y*volume*2**31).long()
        torchaudio.save(sinewave_filepath, y, sr)
        self.assertTrue(os.path.isfile(sinewave_filepath))

if __name__ == '__main__':
    unittest.main()
