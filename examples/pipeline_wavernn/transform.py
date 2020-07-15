import librosa
import torch
import torch.nn as nn


class linear_to_mel(nn.Module):
    def __init__(self, sample_rate, n_fft, n_mels, fmin):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.fmin = fmin

    def forward(self, specgram):
        specgram = librosa.feature.melspectrogram(
            S=specgram.squeeze(0).numpy(),
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
        )
        return torch.from_numpy(specgram)


class specgram_normalize(nn.Module):
    r"""Normalize the spectrogram with a minimum db value
    """

    def __init__(self, min_level_db):
        super().__init__()
        self.min_level_db = min_level_db

    def forward(self, specgram):
        specgram = 20 * torch.log10(torch.clamp(specgram, min=1e-5))
        return torch.clamp((self.min_level_db - specgram) / self.min_level_db, min=0, max=1)
