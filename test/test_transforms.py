import torch
import torchaudio
import torchaudio.transforms as transforms
import numpy as np
import unittest

STEAM_TRAIN = "assets/steam-train-whistle-daniel_simon.mp3"

class Tester(unittest.TestCase):

    sr = 16000
    freq = 440
    volume = 0.3
    sig = (torch.cos(2*np.pi*torch.arange(0, 4*sr) * freq/sr)).float()
    sig.unsqueeze_(1)
    sig = (sig*volume*2**31).long()

    def test_scale(self):

        audio_orig = self.sig.clone()
        result = transforms.Scale()(audio_orig)
        self.assertTrue(result.min() >= -1. and result.max() <= 1.,
                        "min: {}, max: {}".format(result.min(), result.max()))

        maxminmax = np.abs([audio_orig.min(), audio_orig.max()]).max().astype(np.float)
        result = transforms.Scale(factor=maxminmax)(audio_orig)
        self.assertTrue((result.min() == -1. or result.max() == 1.) and
                        result.min() >= -1. and result.max() <= 1.,
                        "min: {}, max: {}".format(result.min(), result.max()))

    def test_pad_trim(self):

        audio_orig = self.sig.clone()
        length_orig = audio_orig.size(0)
        length_new = int(length_orig * 1.2)

        result = transforms.PadTrim(max_len=length_new)(audio_orig)

        self.assertTrue(result.size(0) == length_new,
                        "old size: {}, new size: {}".format(audio_orig.size(0), result.size(0)))

        audio_orig = self.sig.clone()
        length_orig = audio_orig.size(0)
        length_new = int(length_orig * 0.8)

        result = transforms.PadTrim(max_len=length_new)(audio_orig)

        self.assertTrue(result.size(0) == length_new,
                        "old size: {}, new size: {}".format(audio_orig.size(0), result.size(0)))


    def test_downmix_mono(self):
        
        audio_L = self.sig.clone()
        audio_R = self.sig.clone()
        R_idx = int(audio_R.size(0) * 0.1)
        audio_R = torch.cat((audio_R[R_idx:], audio_R[:R_idx]))

        audio_Stereo = torch.cat((audio_L, audio_R), dim=1)

        self.assertTrue(audio_Stereo.size(1) == 2)

        result = transforms.DownmixMono()(audio_Stereo)

        self.assertTrue(result.size(1) == 1)

    def test_compose(self):

        audio_orig = self.sig.clone()
        length_orig = audio_orig.size(0)
        length_new = int(length_orig * 1.2)
        maxminmax = np.abs([audio_orig.min(), audio_orig.max()]).max().astype(np.float)

        tset = (transforms.Scale(factor=maxminmax),
                transforms.PadTrim(max_len=length_new))
        result = transforms.Compose(tset)(audio_orig)

        self.assertTrue(np.abs([result.min(), result.max()]).max() == 1.)

        self.assertTrue(result.size(0) == length_new)


if __name__ == '__main__':
    unittest.main()
