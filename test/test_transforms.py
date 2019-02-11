from __future__ import print_function
import os
import torch
import torchaudio
import torchaudio.transforms as transforms
import numpy as np
import unittest


class Tester(unittest.TestCase):

    # create a sinewave signal for testing
    sr = 16000
    freq = 440
    volume = .3
    sig = (torch.cos(2 * np.pi * torch.arange(0, 4 * sr).float() * freq / sr))
    sig.unsqueeze_(1)  # (64000, 1)
    sig = (sig * volume * 2**31).long()
    # file for stereo stft test
    test_dirpath = os.path.dirname(os.path.realpath(__file__))
    test_filepath = os.path.join(test_dirpath, "assets",
                                 "steam-train-whistle-daniel_simon.mp3")

    def test_scale(self):

        audio_orig = self.sig.clone()
        result = transforms.Scale()(audio_orig)
        self.assertTrue(result.min() >= -1. and result.max() <= 1.)

        maxminmax = np.abs(
            [audio_orig.min(), audio_orig.max()]).max().astype(np.float)
        result = transforms.Scale(factor=maxminmax)(audio_orig)
        self.assertTrue((result.min() == -1. or result.max() == 1.) and
                        result.min() >= -1. and result.max() <= 1.)

        repr_test = transforms.Scale()
        self.assertTrue(repr_test.__repr__())

    def test_pad_trim(self):

        audio_orig = self.sig.clone()
        length_orig = audio_orig.size(0)
        length_new = int(length_orig * 1.2)

        result = transforms.PadTrim(max_len=length_new, channels_first=False)(audio_orig)
        self.assertEqual(result.size(0), length_new)

        result = transforms.PadTrim(max_len=length_new, channels_first=True)(audio_orig.transpose(0, 1))
        self.assertEqual(result.size(1), length_new)

        audio_orig = self.sig.clone()
        length_orig = audio_orig.size(0)
        length_new = int(length_orig * 0.8)

        result = transforms.PadTrim(max_len=length_new, channels_first=False)(audio_orig)

        self.assertEqual(result.size(0), length_new)

        repr_test = transforms.PadTrim(max_len=length_new, channels_first=False)
        self.assertTrue(repr_test.__repr__())

    def test_downmix_mono(self):

        audio_L = self.sig.clone()
        audio_R = self.sig.clone()
        R_idx = int(audio_R.size(0) * 0.1)
        audio_R = torch.cat((audio_R[R_idx:], audio_R[:R_idx]))

        audio_Stereo = torch.cat((audio_L, audio_R), dim=1)

        self.assertTrue(audio_Stereo.size(1) == 2)

        result = transforms.DownmixMono(channels_first=False)(audio_Stereo)

        self.assertTrue(result.size(1) == 1)

        repr_test = transforms.DownmixMono(channels_first=False)
        self.assertTrue(repr_test.__repr__())

    def test_lc2cl(self):

        audio = self.sig.clone()
        result = transforms.LC2CL()(audio)
        self.assertTrue(result.size()[::-1] == audio.size())

        repr_test = transforms.LC2CL()
        self.assertTrue(repr_test.__repr__())

    def test_compose(self):

        audio_orig = self.sig.clone()
        length_orig = audio_orig.size(0)
        length_new = int(length_orig * 1.2)
        maxminmax = np.abs(
            [audio_orig.min(), audio_orig.max()]).max().astype(np.float)

        tset = (transforms.Scale(factor=maxminmax),
                transforms.PadTrim(max_len=length_new, channels_first=False))
        result = transforms.Compose(tset)(audio_orig)

        self.assertTrue(np.abs([result.min(), result.max()]).max() == 1.)

        self.assertTrue(result.size(0) == length_new)

        repr_test = transforms.Compose(tset)
        self.assertTrue(repr_test.__repr__())

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

        sig = self.sig.clone()
        sig = sig / torch.abs(sig).max()
        self.assertTrue(sig.min() >= -1. and sig.max() <= 1.)

        sig_mu = transforms.MuLawEncoding(quantization_channels)(sig)
        self.assertTrue(sig_mu.min() >= 0. and sig.max() <= quantization_channels)

        sig_exp = transforms.MuLawExpanding(quantization_channels)(sig_mu)
        self.assertTrue(sig_exp.min() >= -1. and sig_exp.max() <= 1.)

        repr_test = transforms.MuLawEncoding(quantization_channels)
        self.assertTrue(repr_test.__repr__())
        repr_test = transforms.MuLawExpanding(quantization_channels)
        self.assertTrue(repr_test.__repr__())

    def test_mel2(self):
        audio_orig = self.sig.clone()  # (16000, 1)
        audio_scaled = transforms.Scale()(audio_orig)  # (16000, 1)
        audio_scaled = transforms.LC2CL()(audio_scaled)  # (1, 16000)
        mel_transform = transforms.MelSpectrogram()
        # check defaults
        spectrogram_torch = mel_transform(audio_scaled)  # (1, 319, 40)
        self.assertTrue(spectrogram_torch.dim() == 3)
        self.assertTrue(spectrogram_torch.le(0.).all())
        self.assertTrue(spectrogram_torch.ge(mel_transform.top_db).all())
        self.assertEqual(spectrogram_torch.size(-1), mel_transform.n_mels)
        # check correctness of filterbank conversion matrix
        self.assertTrue(mel_transform.fm.fb.sum(1).le(1.).all())
        self.assertTrue(mel_transform.fm.fb.sum(1).ge(0.).all())
        # check options
        kwargs = {"window": torch.hamming_window, "pad": 10, "ws": 500, "hop": 125, "n_fft": 800, "n_mels": 50}
        mel_transform2 = transforms.MelSpectrogram(**kwargs)
        spectrogram2_torch = mel_transform2(audio_scaled)  # (1, 506, 50)
        self.assertTrue(spectrogram2_torch.dim() == 3)
        self.assertTrue(spectrogram2_torch.le(0.).all())
        self.assertTrue(spectrogram2_torch.ge(mel_transform.top_db).all())
        self.assertEqual(spectrogram2_torch.size(-1), mel_transform2.n_mels)
        self.assertTrue(mel_transform2.fm.fb.sum(1).le(1.).all())
        self.assertTrue(mel_transform2.fm.fb.sum(1).ge(0.).all())
        # check on multi-channel audio
        x_stereo, sr_stereo = torchaudio.load(self.test_filepath)
        spectrogram_stereo = mel_transform(x_stereo)
        self.assertTrue(spectrogram_stereo.dim() == 3)
        self.assertTrue(spectrogram_stereo.size(0) == 2)
        self.assertTrue(spectrogram_stereo.le(0.).all())
        self.assertTrue(spectrogram_stereo.ge(mel_transform.top_db).all())
        self.assertEqual(spectrogram_stereo.size(-1), mel_transform.n_mels)
        # check filterbank matrix creation
        fb_matrix_transform = transforms.MelScale(n_mels=100, sr=16000, f_max=None, f_min=0., n_stft=400)
        self.assertTrue(fb_matrix_transform.fb.sum(1).le(1.).all())
        self.assertTrue(fb_matrix_transform.fb.sum(1).ge(0.).all())
        self.assertEqual(fb_matrix_transform.fb.size(), (400, 100))

if __name__ == '__main__':
    unittest.main()
