
import torch
import torch.nn as nn
import torchaudio

x, sample_rate = torchaudio.load("steam-train-whistle-daniel_simon.mp3")
print(sample_rate)
print(x.size())
print(x[10000])
print(x.min(), x.max())
print(x.mean(), x.std())

x, sample_rate = torchaudio.load("steam-train-whistle-daniel_simon.mp3",
                                 out=torch.LongTensor())
print(sample_rate)
print(x.size())
print(x[10000])
print(x.min(), x.max())
