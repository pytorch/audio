import librosa
import torch


def linear_to_mel(specgram, sample_rate, n_fft, n_mels, fmin):

    specgram = librosa.feature.melspectrogram(
        S=specgram.squeeze(0).numpy(),
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin
    )
    return torch.from_numpy(specgram)
