import librosa
import torch
import torch.nn as nn


# TODO Replace by torchaudio, once https://github.com/pytorch/audio/pull/593 is resolved
class LinearToMel(nn.Module):
    def __init__(self, sample_rate, n_fft, n_mels, fmin, htk=False, norm="slaney"):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.fmin = fmin
        self.htk = htk
        self.norm = norm

    def forward(self, specgram):
        specgram = librosa.feature.melspectrogram(
            S=specgram.squeeze(0).numpy(),
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            htk=self.htk,
            norm=self.norm,
        )
        return torch.from_numpy(specgram)


class NormalizeDB(nn.Module):
    r"""Normalize the spectrogram with a minimum db value
    """

    def __init__(self, min_level_db, normalization):
        super().__init__()
        self.min_level_db = min_level_db
        self.normalization = normalization

    def forward(self, specgram):
        if self.normalization:
            specgram = 20 * torch.log10(torch.clamp(specgram, min=1e-5))
            return torch.clamp(
                (self.min_level_db - specgram) / self.min_level_db, min=0, max=1
            )
        else:
            return torch.log10(torch.clamp(specgram, min=1e-5))


def normalized_waveform_to_bits(waveform, bits):
    r"""Transform waveform [-1, 1] to label [0, 2 ** bits - 1]
    """

    assert abs(waveform).max() <= 1.0
    waveform = (waveform + 1.0) * (2 ** bits - 1) / 2
    return torch.clamp(waveform, 0, 2 ** bits - 1).int()


def bits_to_normalized_waveform(label, bits):
    r"""Transform label [0, 2 ** bits - 1] to waveform [-1, 1]
    """

    return 2 * label / (2 ** bits - 1.0) - 1.0
