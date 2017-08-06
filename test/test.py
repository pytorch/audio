
import torch
import torch.nn as nn
import torchaudio
import math

steam_train = "assets/steam-train-whistle-daniel_simon.mp3"

x, sample_rate = torchaudio.load(steam_train)
print(sample_rate)
print(x.size())
print(x[10000])
print(x.min(), x.max())
print(x.mean(), x.std())

x, sample_rate = torchaudio.load(steam_train,
                                 out=torch.LongTensor())
print(sample_rate)
print(x.size())
print(x[10000])
print(x.min(), x.max())

sine_wave = "assets/sinewave.wav"
sr = 16000
freq = 440
volume = 0.3

y = (torch.cos(2*math.pi*torch.arange(0, 4*sr) * freq/sr)).float()
y.unsqueeze_(1)
# y is between -1 and 1, so must scale
y = (y*volume*2**31).long()
torchaudio.save(sine_wave, y, sr)
print(y.min(), y.max())
