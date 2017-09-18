import torch
import torchaudio
import torchaudio.transforms as transforms
import numpy as np
import unittest

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
                        print("min: {}, max: {}".format(result.min(), result.max())))

        maxminmax = np.abs([audio_orig.min(), audio_orig.max()]).max().astype(np.float)
        result = transforms.Scale(factor=maxminmax)(audio_orig)
        self.assertTrue((result.min() == -1. or result.max() == 1.) and
                        result.min() >= -1. and result.max() <= 1.,
                        print("min: {}, max: {}".format(result.min(), result.max())))

    def test_pad_trim(self):

        audio_orig = self.sig.clone()
        length_orig = audio_orig.size(0)
        length_new = int(length_orig * 1.2)

        result = transforms.PadTrim(max_len=length_new)(audio_orig)

        self.assertTrue(result.size(0) == length_new,
                        print("old size: {}, new size: {}".format(audio_orig.size(0), result.size(0))))

        audio_orig = self.sig.clone()
        length_orig = audio_orig.size(0)
        length_new = int(length_orig * 0.8)

        result = transforms.PadTrim(max_len=length_new)(audio_orig)

        self.assertTrue(result.size(0) == length_new,
                        print("old size: {}, new size: {}".format(audio_orig.size(0), result.size(0))))


    def test_downmix_mono(self):

        audio_L = self.sig.clone()
        audio_R = self.sig.clone()
        R_idx = int(audio_R.size(0) * 0.1)
        audio_R = torch.cat((audio_R[R_idx:], audio_R[:R_idx]))

        audio_Stereo = torch.cat((audio_L, audio_R), dim=1)

        self.assertTrue(audio_Stereo.size(1) == 2)

        result = transforms.DownmixMono()(audio_Stereo)

        self.assertTrue(result.size(1) == 1)

    def test_lc2cl(self):

        audio = self.sig.clone()
        result = transforms.LC2CL()(audio)
        self.assertTrue(result.size()[::-1] == audio.size())

    def test_mel(self):

        audio = self.sig.clone()
        audio = transforms.Scale()(audio)
        self.assertTrue(len(audio.size()) == 2)
        result = transforms.MEL()(audio)
        self.assertTrue(len(result.size()) == 3)
        result = transforms.BLC2CBL()(result)
        self.assertTrue(len(result.size()) == 3)

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

    def test_mu_law_companding(self):

        sig = self.sig.clone()

        quantization_channels = 256
        sig = self.sig.numpy()
        sig = sig / np.abs(sig).max()
        self.assertTrue(sig.min() >= -1. and sig.max() <= 1.)

        sig_mu = transforms.MuLawEncoding(quantization_channels)(sig)
        self.assertTrue(sig_mu.min() >= 0. and sig.max() <= quantization_channels)

        sig_exp = transforms.MuLawExpanding(quantization_channels)(sig_mu)
        self.assertTrue(sig_exp.min() >= -1. and sig_exp.max() <= 1.)

        #diff = sig - sig_exp
        #mse = np.linalg.norm(diff) / diff.shape[0]
        #self.assertTrue(mse, np.isclose(mse, 0., atol=1e-4)) # not always true

        sig = self.sig.clone()
        sig = sig / torch.abs(sig).max()
        self.assertTrue(sig.min() >= -1. and sig.max() <= 1.)

        sig_mu = transforms.MuLawEncoding(quantization_channels)(sig)
        self.assertTrue(sig_mu.min() >= 0. and sig.max() <= quantization_channels)

        sig_exp = transforms.MuLawExpanding(quantization_channels)(sig_mu)
        self.assertTrue(sig_exp.min() >= -1. and sig_exp.max() <= 1.)

if __name__ == '__main__':
    unittest.main()
