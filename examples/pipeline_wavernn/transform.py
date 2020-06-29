import librosa
import numpy as np
import torch


class Transform():
    def __init__(self,
                 sample_rate,
                 n_fft,
                 hop_length,
                 win_length,
                 num_mels,
                 fmin,
                 min_level_db):

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.num_mels = num_mels
        self.fmin = fmin
        self.min_level_db = min_level_db

    def load(self, path):
        waveform = librosa.load(path, sr=self.sample_rate)[0]
        return torch.from_numpy(waveform)

    def stft(self, y):
        return librosa.stft(
            y=y,
            n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)

    def linear_to_mel(self, spectrogram):
        return librosa.feature.melspectrogram(
            S=spectrogram, sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.num_mels, fmin=self.fmin)

    def normalize(self, S):
        return np.clip((S - self.min_level_db) / - self.min_level_db, 0, 1)

    def denormalize(self, S):
        return (np.clip(S, 0, 1) * -self.min_level_db) + self.min_level_db

    def amp_to_db(self, x):
        return 20 * np.log10(np.maximum(1e-5, x))

    def db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def melspectrogram(self, y):
        D = self.stft(y.numpy())
        S = self.amp_to_db(self.linear_to_mel(np.abs(D)))
        S = self.normalize(S)
        return torch.from_numpy(S).float()

    def mulaw_encode(self, x, mu):
        x = x.numpy()
        mu = mu - 1
        fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
        return torch.from_numpy(np.floor((fx + 1) / 2 * mu + 0.5)).int()
