import math

import torch
import torchaudio


DECIBEL = 2 * 20 * math.log10(torch.iinfo(torch.int16).max)
GAIN = pow(10, 0.05 * DECIBEL)
spectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, n_mels=80, hop_length=160)


def piecewise_linear_log(x):
    x[x > math.e] = torch.log(x[x > math.e])
    x[x <= math.e] = x[x <= math.e] / math.e
    return x
